# Preprocessing and Inverted Index

## Preprocessing

Preprocessing prepares the documents for indexing and ensures consistency across the collection. The following steps are applied:

### Text Normalization

Convert all text to lowercase to avoid case sensitivity and remove special characters, punctuation, and non-alphanumeric symbols.

#### Example:

```plaintext
Input: "Artificial Intelligence is Important!"
Output: "artificial intelligence is important"
```

### Tokenization

Split the normalized text into individual words (tokens).

#### Example:

```plaintext
Input: "artificial intelligence is important"
Output: ["artificial", "intelligence", "is", "important"]
```

### Stopword Removal

Remove common, uninformative words (e.g., "and," "the," "is"). A predefined stopword list is used for this step.

#### Example:

```plaintext
Input: ["artificial", "intelligence", "is", "important"]
Output: ["artificial", "intelligence", "important"]
```

### Lemmatization

Convert words to their base forms (e.g., "running" → "run"). This ensures consistency in word forms across documents.

#### Example:

```plaintext
Input: ["running", "dogs"]
Output: ["run", "dog"]
```

### Final Output

After preprocessing, the cleaned and structured tokens are ready for indexing.

---

## Inverted Index

The **inverted index** is a data structure that maps terms to the documents they appear in, along with their computed TF-IDF scores. This allows for efficient searching and retrieval of relevant documents.

### Structure

The inverted index is implemented as a JSON object. Each term maps to the documents it occurs in, along with its associated TF-IDF weight.

#### Example:

```json
{
  "artificial": {
    "doc1": 0.1234,
    "doc5": 0.4567
  },
  "intelligence": {
    "doc1": 0.2345,
    "doc3": 0.6789
  },
  "important": {
    "doc2": 0.321
  }
}
```

### Construction Process

#### Preprocess the Documents

Each document undergoes the preprocessing steps outlined above: normalization, tokenization, stopword removal, and lemmatization.

#### Calculate Term Frequency (TF)

For each term \( t \) in a document \( d \), count how many times it occurs and normalize it by the total number of terms in \( d \).

#### Example:

```plaintext
Input: Term "artificial" appears 3 times in a document with 100 words.
Output: TF = 3 / 100 = 0.03
```

#### Calculate Inverse Document Frequency (IDF)

Compute the importance of each term across the entire document collection.

#### Example:

```plaintext
Input: Term "artificial" appears in 5 out of 100 documents.
Output: IDF = log10(100 / 5) = 1.3010
```

#### Compute TF-IDF

Multiply the TF and IDF values to calculate the TF-IDF weight for each term in each document.

#### Example:

```plaintext
Input: TF = 0.03, IDF = 1.3010
Output: TF-IDF = 0.03 × 1.3010 = 0.03903
```

#### Store the Inverted Index

Save the results in a JSON file (e.g., `inverted_index.json`) for efficient searching.

#### Example JSON File:

```json
{
  "artificial": {
    "doc1": 0.1234,
    "doc5": 0.4567
  },
  "intelligence": {
    "doc1": 0.2345,
    "doc3": 0.6789
  },
  "important": {
    "doc2": 0.321
  }
}
```

### Benefits of the Inverted Index

The inverted index provides the following advantages:

- **Efficient Retrieval**: Enables fast lookup of relevant documents by precomputing term-document mappings.
- **Scalability**: Handles large document collections effectively due to structured indexing.
- **Reusable Data**: The JSON format allows the index to be reused without requiring recomputation for every query.

## Why We Use Cosine Similarity

Cosine Similarity measures how similar the query is to a document by comparing their TF-IDF vectors. It calculates the angle between the vectors, ignoring their length, which makes it ideal for text comparisons.

### Why It's Useful:

- **Focus on Relevant Content**: Measures similarity based on shared terms, regardless of document or query length.
- **Normalized Scores**: Results are between **0** (no match) and **1** (perfect match).
- **Efficient with Sparse Data**: Works well even when query and document share few terms.

### How It Works:

In simple terms:

- Higher scores mean better matches.
- Scores depend on overlapping terms and their TF-IDF weights.

### Sample questions

"How is AI used in generating images?"
"How does AI help in online shopping?"
"What is generative AI?"
"What is the impact of AI on global jobs?"
"How can AI increase inequality?"
"What is the AI Preparedness Index?"
