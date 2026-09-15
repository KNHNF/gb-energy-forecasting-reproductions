# GB Energy Forecasting: Paper Reproductions

Independent reproductions of three papers from the GB balancing market and
imbalance price forecasting literature, done as groundwork before building
my own MSc dissertation model (forecasting total GB Balancing Mechanism
cost, a target none of these papers model directly).

No published code exists for any of these papers. Each subfolder rebuilds
the method from the paper text, against real Elexon BMRS and Carbon
Intensity data, using [gb-bm-data](https://github.com/KNHNF/gb-bm-data) as
the data layer.

These are reproduction attempts, not replications. Each README states
plainly where the result differs from the paper and why, rather than
adjusting the method until the numbers match.

## The three papers

| Folder | Paper | What it tests |
|---|---|---|
| [lucas-2020-reproduction](lucas-2020-reproduction) | Lucas, A., Pegios, K., Kotsakis, E. and Clarke, D. (2020) 'Price Forecasting for the Balancing Energy Market Using Machine-Learning Regression', *Energies*, 13(20), p. 5420 | The paper's feature set and model comparison, rebuilt from scratch |
| [deng-2023-reproduction](deng-2023-reproduction) | Deng, S., Inekwe, J.N., Smirnov, V., Wait, A. and Wang, C. (2023) 'Machine Learning and Deep Learning Forecasts of Electricity Imbalance Prices', *SSRN* | The headline claim: a seasonal-attention BiLSTM beats a plain BiLSTM baseline on extreme price events |
| [bunn-ganesh-2021-2024-reproduction](bunn-ganesh-2021-2024-reproduction) | Bunn, D.W., Inekwe, J.N. and MacGeehan, D. (2021) 'Analysis of the Fundamental Predictability of Prices in the British Balancing Market', *IEEE Transactions on Power Systems*, 36(2), pp. 1309-1316; Ganesh, V.N. and Bunn, D. (2024) 'Forecasting Imbalance Price Densities With Statistical Methods and Neural Networks', *IEEE Transactions on Energy Markets, Policy and Regulation*, 2(1), pp. 30-39 | Two related papers on the same GB balancing dataset, the second benchmarking directly against the first |

Confidence is not equal across the three. Lucas (2020) is the most direct
reproduction. Deng (2023) is stated as weaker, because the exact
architecture and date range could not be confirmed with high confidence
from the source PDF. Bunn/Ganesh is the most data-constrained, since only
2 of the original 7 to 8 explanatory variables were reproducible from
public sources. Each subfolder's own README explains this in full.

## Why this exists

A forecasting result only means something next to a stated benchmark. These
three reproductions establish what the published literature actually claims
on this market, using data and code I can verify myself, before comparing
my own dissertation model against it.

## Setup

Each subfolder is a self-contained project with its own `pyproject.toml`,
`requirements.txt`, and `LICENSE`. See the subfolder README for exact
reproduction commands.

## References

Bunn, D.W., Inekwe, J.N. and MacGeehan, D. (2021) 'Analysis of the
Fundamental Predictability of Prices in the British Balancing Market',
*IEEE Transactions on Power Systems*, 36(2), pp. 1309-1316. Available at:
https://doi.org/10.1109/tpwrs.2020.3015871

Deng, S., Inekwe, J.N., Smirnov, V., Wait, A. and Wang, C. (2023) 'Machine
Learning and Deep Learning Forecasts of Electricity Imbalance Prices'.
SSRN. Available at: https://ssrn.com/abstract=4475210

Ganesh, V.N. and Bunn, D. (2024) 'Forecasting Imbalance Price Densities
With Statistical Methods and Neural Networks', *IEEE Transactions on
Energy Markets, Policy and Regulation*, 2(1), pp. 30-39. Available at:
https://ieeexplore.ieee.org/document/10177230/

Lucas, A., Pegios, K., Kotsakis, E. and Clarke, D. (2020) 'Price
Forecasting for the Balancing Energy Market Using Machine-Learning
Regression', *Energies*, 13(20), p. 5420. Available at:
https://doi.org/10.3390/en13205420
