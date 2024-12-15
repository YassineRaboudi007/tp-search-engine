const express = require("express");
const fs = require("fs");
const path = require("path");
const { calculateTFIDF, search, loadDocuments } = require("./processing");

const app = express();
app.use(express.json()); // Middleware to parse JSON body
const PORT = 3000;

const documentsFolder = path.join(__dirname, "documents");
const invertedIndexPath = path.join(__dirname, "inverted_index.json");

if (!fs.existsSync(documentsFolder)) {
  fs.mkdirSync(documentsFolder);
}

app.post("/search", (req, res) => {
  const { query } = req.body;
  if (!query) {
    return res.status(400).send("Query parameter 'q' is required.");
  }

  let tfidf = null;

  if (fs.existsSync(invertedIndexPath)) {
    const data = fs.readFileSync(invertedIndexPath, "utf-8");
    tfidf = JSON.parse(data);
  } else {
    const documents = loadDocuments(documentsFolder);
    if (documents.length === 0) {
      return res
        .status(400)
        .send(
          "No documents found. Add some .txt files to the documents folder."
        );
    }

    tfidf = calculateTFIDF(documents);
    fs.writeFileSync(
      invertedIndexPath,
      JSON.stringify(tfidf, null, 2),
      "utf-8"
    );
  }

  const documents = loadDocuments(documentsFolder);
  const results = search(query, tfidf);
  const response = results.map((result) => ({
    id: result.id + 1,
    score: result.score.toFixed(4),
    snippet: documents[result.id].substring(0, 200),
  }));

  res.json(response);
});

if (!fs.existsSync(invertedIndexPath)) {
  const documents = loadDocuments(documentsFolder);
  console.log("documents", documents);
  if (documents.length === 0) {
    return res
      .status(400)
      .send("No documents found. Add some .txt files to the documents folder.");
  }

  console.log("cal tfidf");
  tfidf = calculateTFIDF(documents);
  fs.writeFileSync(invertedIndexPath, JSON.stringify(tfidf, null, 2), "utf-8");
}

app.listen(PORT, () => {
  console.log(`Search API running at http://localhost:${PORT}`);
});
