const fs = require("fs");
const path = require("path");
const stopWords = require("stopword");
const nlp = require("compromise");

// Cache for preprocessed documents
const preprocessCache = new Map();
const documentVectorCache = new Map();

// Pre-compile regex for better performance
const tokenRegex = /\W+/;

function mapToObject(map) {
  const obj = {};
  for (const [key, value] of map.entries()) {
    if (value instanceof Map) {
      obj[key] = mapToObject(value); // Recursively transform nested maps
    } else {
      obj[key] = value; // Directly assign the value
    }
  }
  return obj;
}


function logTimeTaken(startTime, label) {
  const timeTaken = (performance.now() - startTime).toFixed(2);
  console.log(`${label}: ${timeTaken}ms`);
}

function preprocess(text) {
  const startTime = performance.now();
  console.log('\n🔄 Starting preprocessing...');

  // Check cache first
  const cacheKey = typeof text === 'string' ? text : '';
  if (preprocessCache.has(cacheKey)) {
    console.log('✓ Cache hit! Returning preprocessed text');
    logTimeTaken(startTime, '📊 Preprocessing (cached)');
    return preprocessCache.get(cacheKey);
  }

  // Check if input is empty or invalid
  if (!text || typeof text !== 'string') {
    console.log('⚠️ Invalid input detected');
    return [];
  }

  console.log('📝 Processing text...');

  // Create initial document and process in one pass
  const doc = nlp(text);
  doc.normalize()
    .nouns().toSingular()
    .verbs().toPastTense();

  // Process text and create tokens in one pass
  const tokens = doc.text()
    .toLowerCase()
    .split(tokenRegex)
    .filter(token => token.length > 0);

  console.log(`📊 Found ${tokens.length} initial tokens`);

  // Remove stopwords
  const result = stopWords.removeStopwords(tokens);
  console.log(`📊 ${tokens.length - result.length} stopwords removed`);

  // Cache the result
  preprocessCache.set(cacheKey, result);

  logTimeTaken(startTime, '📊 Total preprocessing time');
  return result;
}

function calculateTFIDF(docs) {
  const startTime = performance.now();
  console.log('\n🔄 Starting TF-IDF calculation...');
  console.log(`📚 Processing ${docs.length} documents`);

  const tf = new Map();
  const idf = new Map();
  const N = docs.length;
  const tfidf = new Map();

  // Calculate TF and document frequencies in a single pass
  console.log('📊 Calculating term frequencies...');
  const tfStartTime = performance.now();

  docs.forEach((doc, docId) => {
    if (docId % 100 === 0) {
      console.log(`📝 Processing document ${docId + 1}/${docs.length}`);
    }

    const tokens = preprocess(doc);
    const docLength = tokens.length;
    const termFreq = new Map();
    const uniqueTerms = new Set();

    // Count term frequencies
    tokens.forEach(token => {
      termFreq.set(token, (termFreq.get(token) || 0) + 1);
      uniqueTerms.add(token);
    });

    // Normalize term frequencies
    termFreq.forEach((freq, term) => {
      termFreq.set(term, freq / docLength);
    });

    tf.set(docId, termFreq);

    // Update document frequencies
    uniqueTerms.forEach(term => {
      idf.set(term, (idf.get(term) || 0) + 1);
    });
  });

  logTimeTaken(tfStartTime, '📊 Term frequency calculation');

  // Calculate IDF values
  console.log('📊 Calculating IDF values...');
  const idfStartTime = performance.now();

  idf.forEach((docFreq, term) => {
    idf.set(term, Math.log(1 + N / docFreq));
  });

  logTimeTaken(idfStartTime, '📊 IDF calculation');

  // Calculate TF-IDF scores
  console.log('📊 Computing final TF-IDF scores...');
  const tfidfStartTime = performance.now();


  tf.forEach((termFreq, docId) => {
    if (docId % 100 === 0) {
      console.log(`📝 Computing TF-IDF for document ${docId + 1}/${docs.length}`);
    }

    const docTfidf = new Map();
    termFreq.forEach((tfValue, term) => {
      docTfidf.set(term, tfValue * idf.get(term));
    });
    tfidf.set(docId, docTfidf);

    // Cache document vector norms
    let norm = 0;
    docTfidf.forEach(value => {
      norm += value * value;
    });
    documentVectorCache.set(docId, Math.sqrt(norm));
  });

  logTimeTaken(tfidfStartTime, '📊 TF-IDF computation');
  logTimeTaken(startTime, '📊 Total TF-IDF calculation time');


  return mapToObject(tfidf);
}



function search(query, tfidf) {
  const startTime = performance.now();
  console.log('\n🔍 Starting search...');
  console.log(`📝 Query: "${query}"`);

  const queryTokens = preprocess(query);
  console.log(`📊 Preprocessed query tokens: ${queryTokens.join(', ')}`);

  const queryVector = new Map();
  let queryNorm = 0;

  // Build query vector and calculate norm in one pass
  console.log('📊 Building query vector...');
  const queryVectorStartTime = performance.now();

  queryTokens.forEach(token => {
    const count = (queryVector.get(token) || 0) + 1;
    queryVector.set(token, count);
  });

  queryVector.forEach((count, term) => {
    const idfValue = Math.log(
      1 + tfidf.size /
      Array.from(tfidf.values()).filter(doc => doc.has(term)).length
    );
    const weight = count * idfValue;
    queryVector.set(term, weight);
    queryNorm += weight * weight;
  });

  queryNorm = Math.sqrt(queryNorm);
  logTimeTaken(queryVectorStartTime, '📊 Query vector construction');

  // Calculate similarity scores using cached document norms
  console.log('📊 Calculating document similarities...');
  const similarityStartTime = performance.now();

  const scores = new Map();
  let documentsProcessed = 0;

  tfidf.forEach((docVector, docId) => {
    documentsProcessed++;
    if (documentsProcessed % 100 === 0) {
      console.log(`📝 Processed ${documentsProcessed}/${tfidf.size} documents`);
    }

    let score = 0;
    const docNorm = documentVectorCache.get(docId);

    queryVector.forEach((queryWeight, term) => {
      if (docVector.has(term)) {
        score += (queryWeight / queryNorm) * (docVector.get(term) / docNorm);
      }
    });

    if (score > 0) {
      scores.set(docId, score);
    }
  });

  logTimeTaken(similarityStartTime, '📊 Similarity calculation');

  // Sort results
  console.log('📊 Sorting results...');
  const sortStartTime = performance.now();

  const results = Array.from(scores.entries())
    .map(([docId, score]) => ({
      id: parseInt(docId, 10),
      score
    }))
    .sort((a, b) => b.score - a.score);

  logTimeTaken(sortStartTime, '📊 Results sorting');

  console.log(`✓ Search complete - found ${results.length} matching documents`);
  logTimeTaken(startTime, '📊 Total search time');

  console.log(results)

  return results;
}

function loadDocuments(folderPath) {
  const startTime = performance.now();
  console.log('\n📚 Loading documents...');
  console.log(`📁 Reading from: ${folderPath}`);

  const files = fs.readdirSync(folderPath)
    .filter(file => path.extname(file) === ".txt");

  console.log(`📊 Found ${files.length} text files`);

  const documents = files.map((file, index) => {
    if (index % 100 === 0) {
      console.log(`📝 Loading file ${index + 1}/${files.length}`);
    }
    const filePath = path.join(folderPath, file);
    return fs.readFileSync(filePath, "utf-8");
  });

  logTimeTaken(startTime, '📊 Document loading time');
  return documents;
}

module.exports = {
  preprocess,
  calculateTFIDF,
  search,
  loadDocuments,
};
