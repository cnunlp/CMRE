
README for CMRE-Dataset
------------------------

### Overview:
The dataset includes 3 files：`train_data.jsonlines`, `test_data_add_clusters-type.jsonlines` and `dev_data.jsonlines`, which correspond to the training, development, and test data.

The annotations include metaphorical relations, boundaries and types of spans that are arguments of metaphorical relations.

### File Structure:

1. The main tags:
    - **sentences**: the token sequence.
    - **sentence_map**: sentence id
    - **clusters**: the boundaries of a pair of spans that belong to a metaphorical relation. For example, in the sentence “[CLS]我们之间的误会涣然冰释[SEP]”，[[8, 11], [1, 7]] indicates the span “涣然冰释”（token index 8 to 11）and the span “我们之间的误会”（token index 0 to 7）form a metaphorical relation.
    - **spans_type**: fine-grained span types：本体(target)，喻体(source)，喻体动作(source action)，喻体属性(source attribute)，喻体部件(source part). For example，`[[8, 11], '喻体动作']` indicates that the span “涣然冰释” (token index 8 to 11) is a source action(“喻体动作”).
    - **label**: 0 or 1, indicating whether the sentence contains metaphorical relations.
    - **clusters_type**: similar to “clusters”, but it indicates the types of metaphorical relations. Metaphorical relations are derived based on the types of spans a metaphorical relation covers.


For more detailed information or updates, you can visit https://github.com/cnunlp/CMRE 