## baseline_v2
очень интересные результаты. при alpha=0.7, и alpha=0.5 результаты хуже, чем при alpha=0.0 (то есть без genre preference вообще). чем больше alpha, тем хуже метрики

❯ cat metrics_alpha_0_7.json                       
{
  "NDCG@10": 0.24432837085895323,
  "Recall@10": 0.16161186668508415,
  "MRR@10": 0.3415523012552301
}%                                                                                                                                                                                           
catboost-movie-recsys/reports/baseline_v2 on  master [?] 
❯ cat metrics_alpha_0_5.json                       
{
  "NDCG@10": 0.30419420951906856,
  "Recall@10": 0.20319822936217086,
  "MRR@10": 0.4176779570963671
}%                                                                                                                                                                                           
catboost-movie-recsys/reports/baseline_v2 on  master [?] 
❯ cat metrics_alpha_0_0.json                       
{
  "NDCG@10": 0.37167871746555514,
  "Recall@10": 0.245586351932103,
  "MRR@10": 0.5061631135020257
}%     

## Текущий вывод

Genre preference в текущем простом виде не улучшает рекомендации по сравнению с popularity baseline_v1, a наоборот, ухудшает ranking-метрики при увеличении его веса.

Baseline v2 diagnostic results:

- `genre_preference_score` has predictive signal, but it is much weaker than `popularity_score_norm`.
- On user level, positives have higher genre score than negatives for ~62.7% of users, while popularity does so for ~89.6%.
- The positive rate by genre score quantiles is non-monotonic: top genre-score bins are not the most relevant.
- Conclusion: naive genre-based personalization is too noisy to dominate ranking.
- Next step: move to a pointwise model.
