# SEP WordBudget1 六组正式Full汇总

六组正式Full与独立CPU审计全部PASS。单位为%。

| Dataset | Victim | Attack | Budget | R@1 | R@5 | R@10 | mAP | Query | NOOP | 违规 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CUHK-PEDES | IRRA | SEP Text-only | 1 Word | 53.48 | 73.31 | 81.08 | 50.02 | 6156 | 9 | 0 |
| CUHK-PEDES | RaSa | SEP Text-only | 1 Word | 57.81 | 75.76 | 82.13 | 53.90 | 6156 | 9 | 0 |
| RSTPReid | IRRA | SEP Text-only | 1 Word | 47.95 | 72.15 | 80.40 | 39.45 | 2000 | 6 | 0 |
| RSTPReid | RaSa | SEP Text-only | 1 Word | 53.55 | 74.95 | 82.45 | 42.55 | 2000 | 6 | 0 |
| ICFG-PEDES | IRRA | SEP Text-only | 1 Word | 53.09 | 70.71 | 77.19 | 32.02 | 19848 | 19 | 0 |
| ICFG-PEDES | RaSa | SEP Text-only | 1 Word | 53.80 | 69.31 | 74.71 | 33.59 | 19848 | 19 | 0 |

## 独立报告

- [CUHK-PEDES × IRRA](cuhk_irra/REPORT.md)。
- [CUHK-PEDES × RaSa](cuhk_rasa/REPORT.md)。
- [RSTPReid × IRRA](rstp_irra/REPORT.md)。
- [RSTPReid × RaSa](rstp_rasa/REPORT.md)。
- [ICFG-PEDES × IRRA](icfg_irra/REPORT.md)。
- [ICFG-PEDES × RaSa](icfg_rasa/REPORT.md)。
