"""Image grouping — cluster multiple photos of the same physical item."""

from __future__ import annotations

import structlog
from thefuzz import fuzz

from image_analyzer.models import EntityType, ImageAnalysisResult, ImageGroup

logger = structlog.get_logger(__name__)


class ImageGrouper:
    """Group analysed images that depict the same physical item.

    The algorithm proceeds in five stages:
    1. Partition results by ``entity_type`` (unclassified items are never grouped).
    2. Pre-merge iPhone ``IMG_E*`` edit pairs (using the ``paired_images`` field).
    3. Exact-match on ``serial_number`` (strongest signal).
    4. Pairwise similarity + agglomerative clustering for the remainder.
    5. Select a representative (highest confidence) for every group.
    """

    def __init__(self, similarity_threshold: float = 0.75) -> None:
        self.similarity_threshold = similarity_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def group_images(
        self, results: list[ImageAnalysisResult]
    ) -> list[ImageGroup]:
        """Return a list of :class:`ImageGroup` objects."""

        # Step 1 — partition by entity type
        partitions: dict[EntityType, list[ImageAnalysisResult]] = {}
        unclassified: list[ImageGroup] = []

        for r in results:
            etype = r.classification.primary_type
            if etype == EntityType.UNCLASSIFIED:
                # Unclassified items are never grouped — each is its own group.
                unclassified.append(
                    ImageGroup(
                        primary=r,
                        members=[],
                        group_confidence=r.classification.confidence,
                    )
                )
            else:
                partitions.setdefault(etype, []).append(r)

        groups: list[ImageGroup] = list(unclassified)

        for _etype, items in partitions.items():
            groups.extend(self._cluster_partition(items))

        return groups

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cluster_partition(
        self, items: list[ImageAnalysisResult]
    ) -> list[ImageGroup]:
        """Cluster a single entity-type partition into groups."""

        # Step 2 — pre-merge IMG_E* pairs
        merged = self._premerge_pairs(items)

        # Step 3 — exact serial_number match
        serial_groups, remaining = self._group_by_serial(merged)

        # Step 4 — agglomerative clustering on remaining
        clustered = self._agglomerative_cluster(remaining)

        all_groups = serial_groups + clustered

        # Step 5 — select representative for each group
        final: list[ImageGroup] = []
        for cluster in all_groups:
            rep = self._select_representative(cluster)
            members = [r for r in cluster if r is not rep]
            final.append(
                ImageGroup(
                    primary=rep,
                    members=members,
                    group_confidence=rep.classification.confidence,
                )
            )
        return final

    def _premerge_pairs(
        self, items: list[ImageAnalysisResult]
    ) -> list[ImageAnalysisResult]:
        """Merge items whose ``paired_images`` reference another item in the list.

        The result with paired_images absorbs its pair — we keep the one with
        the pair list and add the partner's path to ``paired_images`` if not
        already present.  The partner is removed from the returned list.
        """
        path_to_item = {r.image_path: r for r in items}
        consumed: set[str] = set()
        merged: list[ImageAnalysisResult] = []

        for r in items:
            if r.image_path in consumed:
                continue
            if r.paired_images:
                for paired_path in r.paired_images:
                    if paired_path in path_to_item:
                        consumed.add(paired_path)
            merged.append(r)

        return merged

    def _group_by_serial(
        self, items: list[ImageAnalysisResult]
    ) -> tuple[list[list[ImageAnalysisResult]], list[ImageAnalysisResult]]:
        """Group items that share the same non-empty serial number."""
        serial_map: dict[str, list[ImageAnalysisResult]] = {}
        no_serial: list[ImageAnalysisResult] = []

        for r in items:
            sn = self._get_serial(r)
            if sn:
                serial_map.setdefault(sn, []).append(r)
            else:
                no_serial.append(r)

        groups = [v for v in serial_map.values()]
        return groups, no_serial

    def _agglomerative_cluster(
        self, items: list[ImageAnalysisResult]
    ) -> list[list[ImageAnalysisResult]]:
        """Single-linkage agglomerative clustering based on pairwise similarity."""
        if not items:
            return []

        # Start with each item in its own cluster
        clusters: list[list[ImageAnalysisResult]] = [[r] for r in items]

        changed = True
        while changed:
            changed = False
            i = 0
            while i < len(clusters):
                j = i + 1
                while j < len(clusters):
                    if self._clusters_should_merge(clusters[i], clusters[j]):
                        clusters[i].extend(clusters[j])
                        clusters.pop(j)
                        changed = True
                    else:
                        j += 1
                i += 1

        return clusters

    def _clusters_should_merge(
        self,
        a: list[ImageAnalysisResult],
        b: list[ImageAnalysisResult],
    ) -> bool:
        """Return True if any pair across the two clusters exceeds threshold."""
        for ra in a:
            for rb in b:
                if self._compute_similarity(ra, rb) >= self.similarity_threshold:
                    return True
        return False

    def _compute_similarity(
        self, a: ImageAnalysisResult, b: ImageAnalysisResult
    ) -> float:
        """Weighted multi-signal similarity between two analysis results.

        Weights:
            serial_number exact match   0.40
            model_number exact match    0.20
            name fuzzy match            0.20
            perceptual hash similarity  0.15
            manufacturer_brand match    0.05
        """
        # Different entity types -> zero similarity
        if a.classification.primary_type != b.classification.primary_type:
            return 0.0

        score = 0.0

        # --- serial_number (0.40) ---
        sn_a = self._get_serial(a)
        sn_b = self._get_serial(b)
        if sn_a and sn_b and sn_a == sn_b:
            score += 0.40

        # --- model_number (0.20) ---
        mn_a = self._get_model_number(a)
        mn_b = self._get_model_number(b)
        if mn_a and mn_b and mn_a == mn_b:
            score += 0.20

        # --- name fuzzy match (0.20) ---
        name_a = self._get_name(a)
        name_b = self._get_name(b)
        if name_a and name_b:
            ratio = fuzz.ratio(name_a.lower(), name_b.lower()) / 100.0
            score += 0.20 * ratio

        # --- perceptual hash similarity (0.15) ---
        if a.perceptual_hash and b.perceptual_hash:
            phash_sim = self._phash_similarity(a.perceptual_hash, b.perceptual_hash)
            score += 0.15 * phash_sim

        # --- manufacturer_brand (0.05) ---
        brand_a = self._get_brand(a)
        brand_b = self._get_brand(b)
        if brand_a and brand_b and brand_a.lower() == brand_b.lower():
            score += 0.05

        return score

    def _select_representative(
        self, group: list[ImageAnalysisResult]
    ) -> ImageAnalysisResult:
        """Pick the result with the highest classification confidence."""
        return max(group, key=lambda r: r.classification.confidence)

    # ------------------------------------------------------------------
    # Field extractors (handle different ExtractedData types uniformly)
    # ------------------------------------------------------------------

    @staticmethod
    def _get_serial(r: ImageAnalysisResult) -> str | None:
        if r.extracted_data and hasattr(r.extracted_data, "serial_number"):
            return r.extracted_data.serial_number
        return None

    @staticmethod
    def _get_model_number(r: ImageAnalysisResult) -> str | None:
        if r.extracted_data and hasattr(r.extracted_data, "model_number"):
            return r.extracted_data.model_number
        return None

    @staticmethod
    def _get_name(r: ImageAnalysisResult) -> str | None:
        if r.extracted_data and hasattr(r.extracted_data, "name"):
            return r.extracted_data.name
        return None

    @staticmethod
    def _get_brand(r: ImageAnalysisResult) -> str | None:
        if r.extracted_data and hasattr(r.extracted_data, "manufacturer_brand"):
            return r.extracted_data.manufacturer_brand
        # Chemicals use manufacturer_name
        if r.extracted_data and hasattr(r.extracted_data, "manufacturer_name"):
            return r.extracted_data.manufacturer_name
        return None

    @staticmethod
    def _phash_similarity(hash_a: str, hash_b: str) -> float:
        """Compare two hex-encoded perceptual hashes via normalised Hamming distance.

        Returns 1.0 for identical hashes and approaches 0.0 as they diverge.
        """
        try:
            int_a = int(hash_a, 16)
            int_b = int(hash_b, 16)
        except ValueError:
            return 0.0

        xor = int_a ^ int_b
        bit_length = max(int_a.bit_length(), int_b.bit_length(), 1)
        hamming = bin(xor).count("1")
        return 1.0 - (hamming / bit_length)
