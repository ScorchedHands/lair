# Prefill chain-group and matmul-route tuning analysis 2026-11-26T08-10-33Z

Source sweep: `reports/2026-11-26T03-12-26Z-optimize-prefill-chain-group-widths`.
Host posture: threads=32, rowWidth=64, lengths 128/256, kernels gemv/tiled.
Method: medians/min/max are parsed from completed same-session stage samples and available chain throughput samples. The interrupted matmul-throughput tail was not used for a constant change because route widths above zero were already indistinguishable in the measured dense-stage arms.

## Shared-dot chain-group attention_kernel_ms medians
| shape | arm | median [min,max] |
| --- | --- | ---: |
| gemv 128 | auto8 | 107.27 [105.60,109.14] |
| gemv 128 | individual dots off | 120.78 [119.41,122.49] |
| gemv 128 | pinned 2 | 125.41 [124.08,125.74] |
| gemv 128 | pinned 4 | 108.35 [106.37,111.66] |
| gemv 128 | pinned 8 | 106.97 [105.61,107.97] |

| gemv 256 | auto8 | 548.03 [528.64,560.27] |
| gemv 256 | individual dots off | 609.93 [579.62,618.19] |
| gemv 256 | pinned 2 | 616.44 [596.69,646.29] |
| gemv 256 | pinned 4 | 538.86 [520.07,571.60] |
| gemv 256 | pinned 8 | 527.83 [510.01,531.99] |

| tiled 128 | auto8 | 143.63 [132.31,145.93] |
| tiled 128 | individual dots off | 156.06 [152.08,162.15] |
| tiled 128 | pinned 2 | 179.69 [178.70,189.77] |
| tiled 128 | pinned 4 | 138.97 [134.49,160.68] |
| tiled 128 | pinned 8 | 130.71 [129.02,147.17] |

| tiled 256 | auto8 | 528.13 [500.40,535.76] |
| tiled 256 | individual dots off | 606.71 [601.78,619.50] |
| tiled 256 | pinned 2 | 605.10 [572.30,606.74] |
| tiled 256 | pinned 4 | 558.33 [508.27,558.39] |
| tiled 256 | pinned 8 | 505.85 [489.48,519.52] |

## Shared-dot chain-group total_ms medians
| shape | arm | median [min,max] |
| --- | --- | ---: |
| gemv 128 | auto8 | 1167.23 [1166.09,1178.27] |
| gemv 128 | individual dots off | 1192.19 [1148.39,1196.94] |
| gemv 128 | pinned 2 | 1190.49 [1185.23,1194.41] |
| gemv 128 | pinned 4 | 1181.15 [1155.19,1185.41] |
| gemv 128 | pinned 8 | 1170.45 [1166.96,1187.90] |

| gemv 256 | auto8 | 2761.00 [2751.41,2766.37] |
| gemv 256 | individual dots off | 2828.15 [2799.81,2848.61] |
| gemv 256 | pinned 2 | 2854.13 [2840.99,2883.74] |
| gemv 256 | pinned 4 | 2766.10 [2733.51,2821.88] |
| gemv 256 | pinned 8 | 2754.09 [2752.76,2757.40] |

| tiled 128 | auto8 | 1254.60 [1227.64,1278.43] |
| tiled 128 | individual dots off | 1260.97 [1246.79,1273.88] |
| tiled 128 | pinned 2 | 1304.39 [1223.46,1332.48] |
| tiled 128 | pinned 4 | 1247.35 [1240.33,1276.80] |
| tiled 128 | pinned 8 | 1213.39 [1201.97,1219.35] |

| tiled 256 | auto8 | 2736.58 [2661.66,2749.76] |
| tiled 256 | individual dots off | 2835.05 [2751.43,2858.05] |
| tiled 256 | pinned 2 | 2823.35 [2643.97,2832.39] |
| tiled 256 | pinned 4 | 2739.66 [2724.66,2743.89] |
| tiled 256 | pinned 8 | 2681.49 [2612.68,2705.71] |

## Shared-dot chain-group throughput tok_s medians
| shape | arm | median [min,max] |
| --- | --- | ---: |
| gemv 128 | auto8 | 112.27 [112.18,112.36] |
| gemv 128 | individual dots off | 106.93 [104.58,109.28] |
| gemv 128 | pinned 2 | 109.19 [106.53,111.84] |
| gemv 128 | pinned 4 | 109.04 [106.53,111.55] |
| gemv 128 | pinned 8 | 111.47 [111.08,111.85] |

| gemv 256 | auto8 | 94.39 [93.30,95.47] |
| gemv 256 | individual dots off | 91.73 [90.37,93.09] |
| gemv 256 | pinned 2 | 89.02 [86.31,91.72] |
| gemv 256 | pinned 4 | 91.66 [88.63,94.68] |
| gemv 256 | pinned 8 | 93.02 [90.57,95.46] |

| tiled 128 | auto8 | 106.31 [105.60,107.02] |
| tiled 128 | individual dots off | 105.08 [104.08,106.08] |
| tiled 128 | pinned 2 | 104.12 [99.74,108.50] |
| tiled 128 | pinned 4 | 102.30 [100.12,104.47] |
| tiled 128 | pinned 8 | 106.81 [105.59,108.03] |

| tiled 256 | auto8 | 92.06 [91.89,92.23] |
| tiled 256 | individual dots off | 95.03 [94.40,95.65] |
| tiled 256 | pinned 2 | 92.38 [89.09,95.67] |
| tiled 256 | pinned 4 | 92.67 [89.71,95.63] |
| tiled 256 | pinned 8 | 97.86 [97.58,98.13] |

## Matmul-route gate_up_ms medians (shared dots on)
| shape | arm | median [min,max] |
| --- | --- | ---: |
| gemv 128 | route auto8 | 404.85 [404.77,416.77] |
| gemv 128 | route off | 409.38 [400.58,414.77] |
| gemv 128 | route pinned 4 | 396.01 [387.74,409.79] |
| gemv 128 | route pinned 8 | 405.46 [401.08,416.58] |
| gemv 128 | route pinned 16 | 400.67 [399.25,403.92] |

| gemv 256 | route auto8 | 820.63 [819.68,832.85] |
| gemv 256 | route off | 826.58 [812.94,828.88] |
| gemv 256 | route pinned 4 | 798.25 [788.66,832.16] |
| gemv 256 | route pinned 8 | 822.17 [810.59,831.41] |
| gemv 256 | route pinned 16 | 814.83 [802.93,819.19] |

| tiled 128 | route auto8 | 403.43 [353.99,418.05] |
| tiled 128 | route off | 404.76 [353.64,408.44] |
| tiled 128 | route pinned 4 | 415.15 [400.39,424.56] |
| tiled 128 | route pinned 8 | 407.92 [407.46,419.70] |
| tiled 128 | route pinned 16 | 409.59 [403.24,416.26] |

| tiled 256 | route auto8 | 825.11 [717.26,836.69] |
| tiled 256 | route off | 814.21 [723.77,829.86] |
| tiled 256 | route pinned 4 | 834.28 [830.97,835.90] |
| tiled 256 | route pinned 8 | 816.80 [814.07,842.18] |
| tiled 256 | route pinned 16 | 812.27 [807.07,830.54] |

## Matmul-route ffn_down_ms medians (shared dots on)
| shape | arm | median [min,max] |
| --- | --- | ---: |
| gemv 128 | route auto8 | 363.79 [361.95,380.50] |
| gemv 128 | route off | 405.50 [397.21,407.29] |
| gemv 128 | route pinned 4 | 354.05 [350.36,365.64] |
| gemv 128 | route pinned 8 | 359.98 [352.09,386.74] |
| gemv 128 | route pinned 16 | 351.01 [346.90,359.86] |

| gemv 256 | route auto8 | 721.93 [719.32,722.21] |
| gemv 256 | route off | 789.16 [784.57,812.11] |
| gemv 256 | route pinned 4 | 711.04 [693.03,727.54] |
| gemv 256 | route pinned 8 | 705.74 [703.79,758.40] |
| gemv 256 | route pinned 16 | 702.06 [701.66,709.55] |

| tiled 128 | route auto8 | 354.99 [343.24,366.51] |
| tiled 128 | route off | 395.85 [382.20,405.84] |
| tiled 128 | route pinned 4 | 345.67 [344.74,361.23] |
| tiled 128 | route pinned 8 | 356.23 [338.76,379.58] |
| tiled 128 | route pinned 16 | 349.00 [340.68,354.75] |

| tiled 256 | route auto8 | 716.43 [693.73,739.34] |
| tiled 256 | route off | 791.25 [788.74,804.91] |
| tiled 256 | route pinned 4 | 717.39 [712.40,743.51] |
| tiled 256 | route pinned 8 | 719.03 [677.14,762.27] |
| tiled 256 | route pinned 16 | 708.92 [699.62,720.72] |

## Matmul-route total_ms medians (shared dots on)
| shape | arm | median [min,max] |
| --- | --- | ---: |
| gemv 128 | route auto8 | 1222.02 [1210.77,1228.97] |
| gemv 128 | route off | 1254.55 [1249.19,1257.13] |
| gemv 128 | route pinned 4 | 1189.88 [1171.49,1217.23] |
| gemv 128 | route pinned 8 | 1207.22 [1193.91,1248.91] |
| gemv 128 | route pinned 16 | 1200.25 [1187.40,1200.35] |

| gemv 256 | route auto8 | 2948.17 [2920.88,2950.48] |
| gemv 256 | route off | 3002.66 [2987.28,3003.29] |
| gemv 256 | route pinned 4 | 2885.49 [2837.32,2967.40] |
| gemv 256 | route pinned 8 | 2939.90 [2892.97,3020.31] |
| gemv 256 | route pinned 16 | 2900.48 [2877.73,2921.48] |

| tiled 128 | route auto8 | 1288.54 [1258.99,1326.64] |
| tiled 128 | route off | 1323.59 [1312.74,1352.91] |
| tiled 128 | route pinned 4 | 1295.39 [1285.13,1329.73] |
| tiled 128 | route pinned 8 | 1315.47 [1286.00,1369.21] |
| tiled 128 | route pinned 16 | 1291.31 [1269.08,1342.56] |

| tiled 256 | route auto8 | 2886.97 [2848.26,2947.92] |
| tiled 256 | route off | 2947.74 [2899.57,3085.85] |
| tiled 256 | route pinned 4 | 2940.77 [2921.50,3000.52] |
| tiled 256 | route pinned 8 | 2931.90 [2874.44,2973.03] |
| tiled 256 | route pinned 16 | 2875.17 [2872.14,2891.21] |

## Decision

- Keep `SharedInputDotChainGroup = 8` with vector posture. Individual-dot off and narrower pinned groups do not beat the current auto value cleanly across all measured shapes; the clean signal is batched shared-K dots versus individual dots, already captured by the completed attention batch-dot item.
- Keep `MatmulRouteBatchWidth = 8` with vector posture. The hybrid route remains clearly better than disabled for dense stage substages, but pinned widths 4/8/16 do not separate cleanly enough across gemv/tiled length128/256 windows to justify changing the auto constant.
- No production constants or YAML wiring were changed by this analysis.
