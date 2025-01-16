const express = require("express");

const $ = require("lodash")
const fs = require("fs");
const path = require("path");
const { calculateTFIDF, search, loadDocuments } = require("./processing");
const schedule = require('node-schedule');
const searchEngine = require('./processing')

const app = express();
app.use(express.json());
app.use(express.static('public'));
const PORT = 3000;

const documentsFolder = path.join(__dirname, "documents");
const invertedIndexPath = path.join(__dirname, "inverted_index.json");

if (!fs.existsSync(documentsFolder)) {
  fs.mkdirSync(documentsFolder);
}

// Function to update the index
function updateIndex() {
  console.log("Updating search index...");
  const startTime = Date.now();

  const documents = loadDocuments(documentsFolder);
  if (documents.length > 0) {
    const tfidf = calculateTFIDF(documents);
    fs.writeFileSync(invertedIndexPath, JSON.stringify(tfidf, null, 2), "utf-8");

    const duration = Date.now() - startTime;
    console.log(`Index updated successfully in ${duration}ms`);
  }
}

// Schedule index updates every hour
schedule.scheduleJob('0 * * * *', updateIndex);

// Initial index creation if it doesn't exist
if (!fs.existsSync(invertedIndexPath)) {
  updateIndex();
}

app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.post("/search", (req, res) => {
  const { query } = req.body;
  if (!query) {
    return res.status(400).send("Query parameter is required.");
  }

  try {
    console.log(`\nProcessing search query: "${query}"`);

    const data = fs.readFileSync(invertedIndexPath, "utf-8");
    const tfidf = JSON.parse(data);
    const documents = loadDocuments(documentsFolder);
    const results = search(query, tfidf);

    console.log(`Found ${results.length} results`);

    const response = results.map((result) => {
      const snippet = documents[result.id].substring(0, 200);
      console.log(`\nDocument ${result.id + 1}:`);
      console.log(`Score: ${result.score.toFixed(4)}`);
      console.log(`Snippet: ${snippet.slice(0, 50)}...`);

      return {
        id: result.id + 1,
        score: result.score.toFixed(4),
        snippet: snippet,
        title: `Document ${result.id + 1}`,
      };
    });

    res.json(response);
  } catch (error) {
    console.error('Search error:', error);
    res.status(500).send("Error processing search request");
  }
});

app.listen(PORT, () => {
  console.log(`Search API running at http://localhost:${PORT}`);
  updateIndex();
});
