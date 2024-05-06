## Method

Our approach ultimately relies on tuning of LLM for binary classification task while also including information from wiki-data graph domain in LLM pipeline. The representations for target prediction on question-answer pair is acquired by addressing the last hidden layer representation of the $\texttt{[CLS]}$ token of the model. 

According to the nature of task it is obvious that amongst candidate answers only one of them is correct, however the amount of the candidate answers for single question is not known beforehand. During inference we utilize knowledge about only one candidate answer being right and select the most probable answer to be correct according to model scores. This naturally allows to use model trained for classification target for ranking top-1 candidate answer.

![image](https://github.com/shredder67/text-graph/assets/78615928/c40881da-4da4-4e8d-ac7d-d73c3b8abd99)


<br>

---

### Dataset

For our research, we utilized the [TextGraphs17-shared-task](https://github.com/uhh-lt/TextGraphs17-shared-task/tree/main/data/tsv) dataset, consisting of 37,672 question-answer pairs annotated with Wikidata entities. This dataset includes 10 different types of data, notably entities from Wikidata mentioned in both the answer and the corresponding question, as well as a shortest-path graph for each $\texttt{<question,} \texttt{candidate} \texttt{answer>}$ pair.

<br>

---

### Evaluation metrics

During training and evaluation of our models we use metrics same as ones present in the workshop leaderboard, which include $\textbf{accuracy, precision, recall}$  and $\textbf{F1-score}$. It is important to note that accuracy here is quite uninformative due to the dataset's imbalance, with incorrect answers constituting 90\% of the data.

![image](https://github.com/shredder67/text-graph/assets/78615928/71c1abe9-94ac-4d3a-b9b9-a5bf0abd89d9)


<br>

<br>

<br>

<br>

---

- что сказал слепой человек, когда зашел в бар?

- всем привет, кого не видел
