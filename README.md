# Information retrieval coursework

Three group assignments from an information retrieval course (course number 5180, autumn 2022), written in Python. All three work on the TIME test collection: 423 TIME magazine articles from 1963 that come with 83 queries and relevance judgements.

| Part | Topic | Main files |
|---|---|---|
| 1 | Positional inverted index and Boolean AND queries | `index.py` |
| 2 | TF-IDF ranked retrieval: exact top-k and three approximate methods (champion lists, index elimination, cluster pruning) | `assignment2_index.py`, `performance.ipynb` |
| 3 | Rocchio relevance feedback with precision, recall and average precision per iteration; a Lucene index built through PyLucene | `Group_Assignment_3/index.py`, `Group_Assignment_3/practice5.ipynb` |

The indexing and ranking code is written directly on Python lists and dictionaries. Parts 1 and 2 use only the standard library. In part 3 Lucene is used to build an index, but the ranking and the feedback loop run on the hand-built vectors, with NumPy for the vector arithmetic.

The repository was called `5180GroupAssignment1` before it was renamed to `information-retrieval-coursework`.

## Running

```
git clone https://github.com/shubhroses/information-retrieval-coursework.git
cd information-retrieval-coursework
python3 index.py                       # part 1
python3 assignment2_index.py           # part 2
python3 Group_Assignment_3/index.py    # part 3, needs NumPy and PyLucene
```

Run the scripts from the repository root, because they open their data files with relative paths. The notebooks under `Group_Assignment_3/` are the exception: their paths are relative to the folder each notebook is in. Parts 1 and 2 need nothing beyond Python 3 and run unchanged on Python 3.10, 3.13 and 3.14. There is no requirements file.

## Part 1: positional index and AND queries

`index.py` reads every file in `collection/`, removes everything except letters and whitespace, lowercases the words and builds a positional index of the form `{term: [(doc_id, [positions]), ...]}`. Document ids are assigned in the order in which `os.listdir` returns the files. `and_query` intersects the posting lists of the query terms with a two-pointer merge: it merges the lists in pairs, then the results in pairs, until one list is left. Only document ids are intersected; the stored positions are not used by the query.

Running the script indexes the 423 files (22,494 distinct terms), runs the five sample AND queries at the bottom of the file and appends the build time, the matching file names and the query times to `output.txt`.

## Part 2: TF-IDF retrieval, exact and approximate

`assignment2_index.py` adds weights to the index: `idf = log10(N / df)` for each term and `1 + log10(tf)` for each posting, after dropping the 25 words in `stop-list.txt`. Documents and queries become dense TF-IDF vectors over the 22,469-term vocabulary and are compared by cosine similarity.

- `exact_query(terms, k)` scores every document and returns the top k.
- `inexact_query_champion(terms, k)` uses champion lists: for each term, the `df // 2 + 1` postings with the highest term-frequency weight, with the idf recomputed from the length of that list. Document vectors are rebuilt from the champion lists alone and then scored as in the exact query.
- `inexact_query_index_elimination(terms, k)` keeps the `n // 2` query terms with the highest idf, where n is the number of query terms found in the vocabulary, and scores every document against that shorter query.
- `inexact_query_cluster_pruning(terms, k)` uses clusters that are set up when the index is built: `setLeaders` picks `int(sqrt(N))` documents at random as leaders (20 here) and attaches every other document to its most similar leader. The query is compared with the leaders, then with the members of their clusters, until k documents have been collected.

Running the script builds the index, the champion lists and the clusters, prints the timings, runs five sample queries through `exact_query` and appends their top-10 lists to `output.txt`. On an Apple M4 it takes about 10 seconds with Python 3.14 and 17 seconds with Python 3.10. Most of that is `setLeaders` comparing every document with every leader in pure Python.

`performance.ipynb` compares the three approximate methods with the exact result for the query `murder trial` at k = 1 to 50, using the exact top-k as ground truth, and plots the F score (beta = 1) against k. It needs Jupyter, NumPy and Matplotlib. Importing `assignment2_index` also runs the five sample queries, because they sit at module level.

## Part 3: Rocchio relevance feedback

`Group_Assignment_3/index.py` parses `time/TIME.ALL` (articles separated by `*TEXT` header lines), splits the text on whitespace, lowercases it, drops the stop words in `time/TIME.STP` (its 26 single-letter entries are skipped) and builds the same kind of TF-IDF vectors as part 2. Punctuation is not stripped here, and the vocabulary has 29,109 terms. It then:

- writes the articles to a Lucene index through PyLucene (`StandardAnalyzer`, stored fields with term positions and offsets);
- implements the Rocchio update `q' = alpha*q + beta*mean(relevant) - gamma*mean(non-relevant)`;
- in `get_stats(qId, k)`, runs six rounds for one of the queries in `time/TIME.QUE`: retrieve the top k by cosine similarity, split them into relevant and non-relevant using `time/TIME.REL`, print precision, recall and average precision (labelled MAP in the output), and update the query vector with alpha = 1.0, beta = 0.75, gamma = 0.15;
- in `get_stats_pseudo(qId, k)`, runs the same loop with pseudo-relevance feedback: the top three results are treated as relevant and the rest of the top k as non-relevant, while the measures are still computed against `time/TIME.REL`.

The Lucene index is written and a reader is opened on it, but `index.py` does not search it. Retrieval and feedback use the vectors built in Python.

Running the script prints the feedback loop for query 6 with k = 9 and writes the Lucene index to `index/` in the current directory. The index writer is opened in Lucene's default create-or-append mode, so every run adds the 423 articles to the index again.

The rest of the folder:

- `practice3.ipynb` does exact top-k retrieval with Lucene on the three-document file `time/test.all`: it indexes the file, reads the document frequencies back through the Lucene index reader, builds TF-IDF vectors and ranks them by cosine similarity for a query typed at a prompt.
- `practice5.ipynb` runs both feedback loops for queries 6, 9 and 12 with k = 50 and plots the three measures per iteration. `output.txt` is a hand-trimmed copy of the `get_stats` output for the same three queries.
- `helper.py`, `indexer.py` and `retriever.py` are the parsers for the TIME file formats and a minimal PyLucene indexer and searcher, written against the three-document files `time/test.*`. `retriever.py` searches the `index/` directory that `indexer.py` writes, so `indexer.py` has to be run first.
- `practice4.ipynb`, `practice_files/` and `testing_functions/` are the notebooks in which the parsers and the first PyLucene calls were tried out.

Part 3 needs NumPy and PyLucene, and `practice5.ipynb` also needs Matplotlib. PyLucene is not on PyPI. It is built from source with JCC, a JDK and a C++ compiler, as described in the [PyLucene build instructions](https://lucene.apache.org/pylucene/install.html). The repository has no Dockerfile for it. In November 2022 the code was run with Lucene 9.1.0 on Java 11 under Debian on aarch64. Those versions come from the metadata of the index files it wrote at the time, which are generated and are no longer kept in the repository.

## The TIME collection

The TIME collection is a public test collection for retrieval experiments: articles from TIME magazine (the article headers are dated 4 January to 27 December 1963), 83 queries and, for each query, the numbers of the articles judged relevant. The University of Glasgow hosts it at http://ir.dcs.gla.ac.uk/resources/test_collections/time/ under the file names `readme`, `time.all`, `time.que`, `time.rel` and `time.stp`, and its index of test collections lists it with 423 documents and 83 queries. `Group_Assignment_3/time/` holds the same set of files:

- `TIME.ALL`: the 423 articles, each introduced by a `*TEXT` header line.
- `TIME.QUE`: the 83 queries.
- `TIME.REL`: the relevance judgements, one line per query.
- `TIME.STP`: 340 stop words.
- `README`: the note that came with the collection, a 1988 email about how the files were prepared.

`test.all`, `test.que`, `test.rel` and `test.stp` in the same folder are not part of the collection. They are three-document files in the same formats, written to test the parsers.

`collection/` holds the articles one per file for parts 1 and 2: `Text-N.txt` is the N-th article of `TIME.ALL`, without its header line, for N = 1 to 422. `Text-423.txt` is empty.

## Other files

- `stop-list.txt` is the 25-word stop list used in part 2.
- `practice_collection/`, `practice2_collection/` and `practice3_collection/` are three- and four-document toy collections used while developing each method.
- `practice.ipynb`, `practice2.ipynb`, `method1.ipynb`, `championList.ipynb`, `clusterPruning.ipynb` and `indexElimination.ipynb` are the step-by-step working for parts 1 and 2, mostly on the toy collections. `new.ipynb` is a one-cell scratch notebook.
- The notebooks were saved from Python 3.9 and 3.10 kernels.
- `output.txt` in the root, the Lucene `index/` directories and `__pycache__/` are generated when the code runs and are listed in `.gitignore`.

## Known issues

This is coursework as it was submitted. The problems below are documented, not fixed, so that the saved outputs still correspond to the code, except where a bullet says otherwise.

- `inexact_query_cluster_pruning` sorts the leaders, and the members of each cluster, by cosine similarity in ascending order, so it starts with the least similar cluster and takes its least similar members first. In the run saved in `performance.ipynb` its F score is 0 for every k up to 43, and the query matches 43 documents. The leaders are chosen at random, so a rerun gives a different curve.
- The outputs saved in `performance.ipynb` are older than the last change to `assignment2_index.py`. When the notebook was run, champion lists held the top 2 postings per term. The code now keeps `df // 2 + 1`, so a rerun gives a different champion-list curve (F at k = 10 is 0.6 with the current code and 0.2 in the saved plot). The index-elimination curve is unchanged.
- Part 3 identifies each article by the number in its `*TEXT` header (17 to 563, with gaps), whereas `TIME.REL` numbers the articles 1 to 425 by their position in the collection. The relevance judgements are therefore matched against the wrong articles, and the precision, recall and MAP values in `Group_Assignment_3/output.txt` and `practice5.ipynb` are not valid effectiveness figures. For query 6, seven of the ten highest-ranked articles are relevant when the numbers in `TIME.REL` are read as positions, but the code counts one, and the feedback step uses the other six as negative examples. Reading the numbers as positions is not a complete fix either: `TIME.REL` goes up to 425 and `TIME.ALL` holds 423 articles, and from about number 378 onwards the judged article appears to sit two places earlier in the file.
- `collection/Text-423.txt` is empty, so parts 1 and 2 index 422 articles and one empty document, and N is 423 in the idf. The last article of `TIME.ALL` is missing from `collection/`.
- `index.py` and `assignment2_index.py` run their sample queries at module level and append to the same `output.txt` in the current directory. The file grows with every run or import and mixes the output of both parts. `index.py` closes the file when it finishes, so calling `and_query` on the imported object raises `ValueError`.
- `print_dict` and `print_doc_list` in `assignment2_index.py` raise `TypeError`, because `.items` is missing its parentheses. `inexact_query_index_elimination` prints the score of every document, and with a one-term query it keeps no terms, so every document scores 0.

## Credits

- Two-person group project; both contributors appear in the commit history.
- The class and method stubs in `index.py` and `Group_Assignment_3/index.py` come from the assignments' starter files.
- `cosine_similarity` is taken from [Mike Housky's Stack Overflow answer](https://stackoverflow.com/a/18424953) (CC BY-SA 3.0), with a check for zero vectors added.
- The PyLucene indexer and searcher (`indexer.py`, `retriever.py`, and the same code in `index.py` and the notebooks) are adapted from the indexer and retriever in David Graus's [PyLucene 4.0 (in 60 seconds) tutorial](https://graus.nu/blog/pylucene-4-0-in-60-seconds-tutorial/). `practice_files/practice.ipynb` links to the video [9.1. Indexing with PyLucene](https://www.youtube.com/watch?v=kTdfhmpPDG8) by Dwaipayan Roy.
- Champion lists, index elimination, cluster pruning and Rocchio feedback are described in Manning, Raghavan and Schütze, [*Introduction to Information Retrieval*](https://nlp.stanford.edu/IR-book/) (Cambridge University Press, 2008), sections 7.1 and 9.1.
- The TIME collection is third-party test data. The article text is TIME magazine's.
