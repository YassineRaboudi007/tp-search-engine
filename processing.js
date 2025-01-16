const fs = require("fs");
const path = require("path");
const stopWords = require("stopword");
const nlp = require("compromise");

function preprocess(text) {
  let doc = nlp(text);
  doc = doc.normalize(); // Normalize text (lowercase, remove punctuation, expand contractions, etc.)
  doc = doc.nouns().toSingular(); // Convert plural nouns to singular
  doc = doc.sentences().toPastTense(); // Convert sentences to past tense

  let lemmatizedText = doc.text();
  let tokens = lemmatizedText.toLowerCase().split(/\W+/); // Lowercase and tokenize
  tokens = stopWords.removeStopwords(tokens); // Remove stopwords

  return tokens;
}

function calculateTFIDF(docs) {
  const tf = {};
  const idf = {};
  const N = docs.length;

  docs.forEach((doc, docId) => {
    const tokens = preprocess(doc);
    tf[docId] = {};

    tokens.forEach((token) => {
      tf[docId][token] = (tf[docId][token] || 0) + 1;
    });

    Object.keys(tf[docId]).forEach((token) => {
      idf[token] = (idf[token] || 0) + 1;
    });
  });

  for (const term in idf) {
    idf[term] = Math.log(N / idf[term]);
  }

  const tfidf = {};

  docs.forEach((doc, docId) => {
    tfidf[docId] = {};
    for (const term in tf[docId]) {
      const tfValue = tf[docId][term];
      const tfidfValue = tfValue * idf[term];
      tfidf[docId][term] = tfidfValue;
    }
  });

  return tfidf;
}

function search(query, tfidf) {
  const queryTokens = preprocess(query); // Apply the same preprocessing to query
  const queryVector = {};

  queryTokens.forEach((token) => {
    queryVector[token] = (queryVector[token] || 0) + 1;
  });

  const queryWeights = {};
  let sumOfSquares = 0;

  for (const term in queryVector) {
    const tfValue = queryVector[term];
    const idfValue = Object.values(tfidf).some((doc) => term in doc)
      ? Math.log(
          Object.keys(tfidf).length /
            Object.values(tfidf).filter((doc) => term in doc).length
        )
      : 0;

    queryWeights[term] = tfValue * idfValue;
    sumOfSquares += queryWeights[term] ** 2;
  }

  const norm = Math.sqrt(sumOfSquares);

  const docScores = {};
  let docNorm = 0;
  for (const docId in tfidf) {
    for (const term in queryWeights) {
      if (tfidf[docId][term]) {
        docNorm += tfidf[docId][term] ** 2;
      }
    }

    docNorm = Math.sqrt(docNorm);

    for (const term in queryWeights) {
      if (tfidf[docId][term]) {
        console.log("queryWeights[term]", queryWeights[term] / norm);
        console.log("tfidf[docId][term] / norm", tfidf[docId][term] / docNorm);
        docScores[docId] =
          (docScores[docId] || 0) +
          (queryWeights[term] / norm) * (tfidf[docId][term] / docNorm);
      }
    }
  }

  return Object.keys(docScores)
    .map((docId) => ({
      id: parseInt(docId, 10),
      score: docScores[docId],
    }))
    .filter((result) => result.score > 0)
    .sort((a, b) => b.score - a.score);
}

function loadDocuments(folderPath) {
  const documents = [];
  const files = fs.readdirSync(folderPath);

  files.forEach((file) => {
    const filePath = path.join(folderPath, file);
    if (fs.statSync(filePath).isFile() && path.extname(file) === ".txt") {
      const content = fs.readFileSync(filePath, "utf-8");
      documents.push(content);
    }
  });

  return documents;
}

module.exports = {
  preprocess,
  calculateTFIDF,
  search,
  loadDocuments,
};
