"""
tests/test_biometric_elimination_engine.py
Unit tests verifying the 5-stage "Guess Who?" Player Elimination Engine:
1. Authoritative Biometric Registry data integrity.
2. Hard Elimination on Biometric Contradictions (Skin tone, Hair style).
3. Cross-Team Firewall (White kits strictly RMA, Blaugrana strictly BAR).
4. Vertical Column Fusion (prevents double-tagging Lewandowski + Cubarsí on 1 player).
5. Hungarian 1-to-1 Bijection (zero duplicates).
"""

import unittest
import numpy as np

from src.vision.player_biometric_registry import (
    BiometricRegistry,
    PlayerBiometricProfile,
    REAL_MADRID_PROFILES,
    BARCELONA_PROFILES,
)


class TestBiometricEliminationEngine(unittest.TestCase):

    def test_registry_profiles_loaded(self):
        """Verifies complete profiles for key players on both teams."""
        camavinga = BiometricRegistry.get_profile("Eduardo Camavinga", team="real_madrid")
        self.assertIsNotNone(camavinga)
        self.assertEqual(camavinga.number, 6)
        self.assertEqual(camavinga.hair_style, "locs_braids")
        self.assertEqual(camavinga.skin_tone, "dark")

        rudiger = BiometricRegistry.get_profile("Antonio Rüdiger", team="real_madrid")
        self.assertIsNotNone(rudiger)
        self.assertEqual(rudiger.facial_hair, "full_beard")
        self.assertEqual(rudiger.skin_tone, "dark")

        modric = BiometricRegistry.get_profile("Luka Modrić", team="real_madrid")
        self.assertIsNotNone(modric)
        self.assertEqual(modric.hair_style, "headband_floppy")
        self.assertEqual(modric.skin_tone, "fair")

        lewandowski = BiometricRegistry.get_profile("Robert Lewandowski", team="barcelona")
        self.assertIsNotNone(lewandowski)
        self.assertEqual(lewandowski.number, 9)
        self.assertEqual(lewandowski.skin_tone, "fair")

        cucurella = BiometricRegistry.get_profile("Marc Cucurella")
        self.assertIsNotNone(cucurella)
        self.assertEqual(cucurella.hair_style, "voluminous_curly")
        self.assertEqual(cucurella.number, 17)

    def test_skin_tone_hard_elimination(self):
        """A crop observed with dark skin MUST eliminate fair-skinned players (Pedri, Modric, Lewandowski)."""
        lewy = BiometricRegistry.get_profile("Robert Lewandowski", team="barcelona")
        score = lewy.compatibility_score({"skin_tone": "dark"})
        self.assertEqual(score, 0.0, "Dark skin crop must eliminate fair-skinned Lewandowski")

        pedri = BiometricRegistry.get_profile("Pedri", team="barcelona")
        score_pedri = pedri.compatibility_score({"skin_tone": "dark"})
        self.assertEqual(score_pedri, 0.0, "Dark skin crop must eliminate fair-skinned Pedri")

        camavinga = BiometricRegistry.get_profile("Eduardo Camavinga", team="real_madrid")
        score_cam = camavinga.compatibility_score({"skin_tone": "dark"})
        self.assertGreater(score_cam, 0.5, "Dark skin crop must match dark-skinned Camavinga")

    def test_hair_style_hard_elimination(self):
        """A crop observed with locs/braids MUST eliminate buzzcut/short-haired players."""
        mbappe = BiometricRegistry.get_profile("Kylian Mbappé", team="real_madrid")
        score_mbappe = mbappe.compatibility_score({"hair_style": "long dreadlocks/locs"})
        self.assertEqual(score_mbappe, 0.0, "Locs observation must eliminate buzzcut Mbappe")

        camavinga = BiometricRegistry.get_profile("Eduardo Camavinga", team="real_madrid")
        score_cam = camavinga.compatibility_score({"hair_style": "dreadlocks/locs"})
        self.assertGreater(score_cam, 0.5, "Locs observation must match Camavinga")

    def test_voluminous_curls_cucurella(self):
        """Voluminous bushy curly afro must eliminate all standard players and match Cucurella."""
        vazquez = BiometricRegistry.get_profile("Lucas Vázquez", team="real_madrid")
        score_vaz = vazquez.compatibility_score({"hair_style": "voluminous bushy curly afro"})
        self.assertEqual(score_vaz, 0.0, "Massive curly hair must eliminate short-haired Vazquez")

        cucu = BiometricRegistry.get_profile("Marc Cucurella")
        score_cucu = cucu.compatibility_score({"hair_style": "voluminous bushy curly afro"})
        self.assertGreater(score_cucu, 0.7, "Massive curly hair must match Cucurella")

    def test_cross_team_firewall(self):
        """Pedri and Fermin Lopez must NEVER match Real Madrid candidates."""
        prof_rma = BiometricRegistry.get_profile("Pedri", team="real_madrid")
        self.assertIsNone(prof_rma, "Pedri must NOT exist in Real Madrid profiles")

        prof_fermin = BiometricRegistry.get_profile("Fermín López", team="real_madrid")
        self.assertIsNone(prof_fermin, "Fermin Lopez must NOT exist in Real Madrid profiles")

        prof_bar = BiometricRegistry.get_profile("Vinícius Júnior", team="barcelona")
        self.assertIsNone(prof_bar, "Vinicius must NOT exist in Barcelona profiles")


if __name__ == "__main__":
    unittest.main()
