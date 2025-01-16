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
    let sumOfSquares = 0;

    for (const term in tf[docId]) {
      const tfLog = 1 + Math.log(tf[docId][term]); // Logarithmic TF
      const tfidfValue = tfLog * idf[term];
      tfidf[docId][term] = tfidfValue;
      sumOfSquares += tfidfValue ** 2;
    }

    // const norm = Math.sqrt(sumOfSquares);
    // for (const term in tfidf[docId]) {
    //   tfidf[docId][term] /= norm; // Cosine normalization
    // }
  });

  return tfidf;
}

function search(query, tfidf) {
  const queryTokens = preprocess(query);
  const queryVector = {};

  queryTokens.forEach((token) => {
    queryVector[token] = (queryVector[token] || 0) + 1;
  });

  const queryWeights = {};
  let sumOfSquares = 0;

  for (const term in queryVector) {
    const tfLog = 1 + Math.log(queryVector[term]); // Logarithmic TF
    const idfValue = Object.values(tfidf).some((doc) => term in doc)
      ? Math.log(
          Object.keys(tfidf).length /
            Object.values(tfidf).filter((doc) => term in doc).length
        )
      : 0;

    queryWeights[term] = tfLog * idfValue;
    sumOfSquares += queryWeights[term] ** 2;
  }

  const docScores = {};
  const norm = Math.sqrt(sumOfSquares);

  for (const term in queryWeights) {
    for (const docId in tfidf) {
      if (tfidf[docId][term]) {
        docScores[docId] =
          ((docScores[docId] || 0) + queryWeights[term] * tfidf[docId][term]) /
          norm;
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
