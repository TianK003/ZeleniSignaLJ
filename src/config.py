"""
Zeleni SignaLJ - Intersection Configuration
=============================================
TLS IDs for the 5 controlled intersections in the Bleiweisova triangle.
All other traffic lights in the network run default fixed-time programs.
"""

# ── Target intersections ──
# 1. Tivolska / Slovenska / Dunajska / Trg OF
# 2. Bleiweisova / Tivolska / Celovska / Gosposvetska
# 3. Slovenska / Gosposvetska / Dalmatinova
# 4. Askerceva / Presernova / Groharjeva
# 5. Askerceva / Zoisova / Slovenska / Barjanska

TS_IDS = [
    "joinedS_5154793231_8093399326_8093399327_8093399328_#7more",
    "joinedS_1951535395_8569909625_8569909627_8569909629_#5more",
    "joinedS_cluster_10946184173_33632882_4083612498_4898978366_#5more_cluster_4898978371_9307230471_9307230472",
    "joinedS_16191121_311397806_476283378_6264081028_#12more",
    "joinedS_8241154017_8241154018_cluster_1632640893_3884437221_3884437224_4312381314_#8more_cluster_8171896855_8171896868_8241143312",
]

# Map from human-readable names to TLS IDs (for logging/eval)
TS_NAMES = {
    TS_IDS[0]: "Kolodvor",
    TS_IDS[1]: "Pivovarna",
    TS_IDS[2]: "Slovenska",
    TS_IDS[3]: "Trzaska",
    TS_IDS[4]: "Askerceva",
}
