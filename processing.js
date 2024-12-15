const fs = require("fs");
const path = require("path");
const stopWords = require("stopword");
const nlp = require("compromise");

function preprocess(text) {
  let doc = nlp(text);
  doc = doc.normalize();
  doc = doc.nouns().toSingular();
  doc = doc.sentences().toPastTense();

  let lemmatizedText = doc.text();
  let tokens = lemmatizedText.split(/\W+/);
  tokens = stopWords.removeStopwords(tokens);

  return tokens;
}

function calculateTFIDF(docs) {
  const tf = {};
  const idf = {};
  const N = docs.length;

  docs.forEach((doc, docId) => {
    const tokens = preprocess(doc);
    tf[docId] = {};

    const uniqueTokens = new Set(tokens);

    tokens.forEach((token) => {
      tf[docId][token] = (tf[docId][token] || 0) + 1;
    });

    uniqueTokens.forEach((token) => {
      idf[token] = (idf[token] || 0) + 1;
    });
  });

  for (const term in idf) {
    idf[term] = Math.log(N / idf[term]);
  }

  const tfidf = {};
  docs.forEach((doc, docId) => {
    for (const term in tf[docId]) {
      if (!tfidf[term]) {
        tfidf[term] = {};
      }
      tfidf[term][docId] = tf[docId][term] * idf[term];
    }
  });

  return tfidf;
}

function cosineSimilarity(vectorA, vectorB) {
  const dotProduct = Object.keys(vectorA).reduce((sum, key) => {
    return sum + vectorA[key] * (vectorB[key] || 0);
  }, 0);

  const magnitudeA = Math.sqrt(
    Object.values(vectorA).reduce((sum, val) => sum + val * val, 0)
  );

  const magnitudeB = Math.sqrt(
    Object.values(vectorB).reduce((sum, val) => sum + val * val, 0)
  );

  return magnitudeA && magnitudeB ? dotProduct / (magnitudeA * magnitudeB) : 0;
}

function search(query, tfidf) {
  const queryTokens = preprocess(query);
  const queryVector = {};

  queryTokens.forEach((token) => {
    queryVector[token] = (queryVector[token] || 0) + 1;
  });

  const docScores = {};

  for (const term in queryVector) {
    if (tfidf[term]) {
      for (const docId in tfidf[term]) {
        const termWeight = queryVector[term] * tfidf[term][docId];
        docScores[docId] = (docScores[docId] || 0) + termWeight;
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
  cosineSimilarity,
  search,
  loadDocuments,
};
