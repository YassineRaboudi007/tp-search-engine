package main

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"time"
)

// Document represents a processed document with its tokens
type Document struct {
	ID     int
	Tokens []string
}

// SearchResult represents a search result with document ID and relevance score
type SearchResult struct {
	ID    int
	Score float64
}

// SearchEngine handles document processing and searching
type SearchEngine struct {
	documents     []Document
	tfidf        map[int]map[string]float64
	idf          map[string]float64
	docNorms     map[int]float64
	preprocessor  *Preprocessor
	mu           sync.RWMutex
	workerPool   chan struct{}
	stopWords    map[string]struct{}
}

// Preprocessor handles text preprocessing
type Preprocessor struct {
	tokenRegex *regexp.Regexp
}

// NewSearchEngine creates a new search engine instance
func NewSearchEngine(maxWorkers int) *SearchEngine {
	return &SearchEngine{
		tfidf:       make(map[int]map[string]float64),
		idf:         make(map[string]float64),
		docNorms:    make(map[int]float64),
		workerPool:  make(chan struct{}, maxWorkers),
		preprocessor: newPreprocessor(),
		stopWords:   loadStopWords(),
	}
}

func newPreprocessor() *Preprocessor {
	return &Preprocessor{
		tokenRegex: regexp.MustCompile(`\W+`),
	}
}

func loadStopWords() map[string]struct{} {
	// Common English stop words - extend as needed
	stopWords := []string{"a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he",
		"in", "is", "it", "its", "of", "on", "that", "the", "to", "was", "were", "will", "with"}
	
	stopWordsMap := make(map[string]struct{}, len(stopWords))
	for _, word := range stopWords {
		stopWordsMap[word] = struct{}{}
	}
	return stopWordsMap
}

func (p *Preprocessor) preprocess(text string) []string {
	// Convert to lowercase
	text = strings.ToLower(text)
	
	// Tokenize
	tokens := p.tokenRegex.Split(text, -1)
	
	// Filter and clean tokens
	validTokens := make([]string, 0, len(tokens))
	for _, token := range tokens {
		if token = strings.TrimSpace(token); len(token) > 0 {
			validTokens = append(validTokens, token)
		}
	}
	
	return validTokens
}

// LoadDocuments loads documents from a directory concurrently
func (se *SearchEngine) LoadDocuments(dirPath string) error {
	start := time.Now()
	fmt.Printf("\n📚 Loading documents from %s...\n", dirPath)

	files, err := filepath.Glob(filepath.Join(dirPath, "*.txt"))
	if err != nil {
		return fmt.Errorf("error finding text files: %v", err)
	}

	fmt.Printf("📊 Found %d text files\n", len(files))

	var wg sync.WaitGroup
	docChan := make(chan Document, len(files))
	errChan := make(chan error, len(files))

	for id, file := range files {
		wg.Add(1)
		se.workerPool <- struct{}{} // Acquire worker

		go func(id int, filePath string) {
			defer wg.Done()
			defer func() { <-se.workerPool }() // Release worker

			content, err := os.ReadFile(filePath)
			if err != nil {
				errChan <- fmt.Errorf("error reading file %s: %v", filePath, err)
				return
			}

			tokens := se.preprocessor.preprocess(string(content))
			docChan <- Document{ID: id, Tokens: tokens}
			
			if id%100 == 0 {
				fmt.Printf("📝 Processed document %d/%d\n", id+1, len(files))
			}
		}(id, file)
	}

	// Close channels when all workers are done
	go func() {
		wg.Wait()
		close(docChan)
		close(errChan)
	}()

	// Collect results and errors
	for doc := range docChan {
		se.documents = append(se.documents, doc)
	}

	for err := range errChan {
		if err != nil {
			return err
		}
	}

	fmt.Printf("✓ Documents loaded in %v\n", time.Since(start))
	return nil
}

// CalculateTFIDF calculates TF-IDF scores concurrently
func (se *SearchEngine) CalculateTFIDF() {
	start := time.Now()
	fmt.Printf("\n🔄 Calculating TF-IDF scores...\n")

	var wg sync.WaitGroup
	termFreqChan := make(chan map[int]map[string]int, len(se.documents))

	// Calculate term frequencies concurrently
	for _, doc := range se.documents {
		wg.Add(1)
		se.workerPool <- struct{}{} // Acquire worker

		go func(doc Document) {
			defer wg.Done()
			defer func() { <-se.workerPool }() // Release worker

			termFreq := make(map[string]int)
			for _, token := range doc.Tokens {
				if _, isStopWord := se.stopWords[token]; !isStopWord {
					termFreq[token]++
				}
			}

			result := make(map[int]map[string]int)
			result[doc.ID] = termFreq
			termFreqChan <- result
		}(doc)
	}

	// Close channel when all workers are done
	go func() {
		wg.Wait()
		close(termFreqChan)
	}()

	// Collect term frequencies and calculate document frequencies
	docFreq := make(map[string]int)
	termFreq := make(map[int]map[string]int)
	
	for tf := range termFreqChan {
		for docID, terms := range tf {
			termFreq[docID] = terms
			for term := range terms {
				docFreq[term]++
			}
		}
	}

	// Calculate IDF and TF-IDF scores
	N := float64(len(se.documents))
	for term, freq := range docFreq {
		se.idf[term] = math.Log1p(N / float64(freq))
	}

	// Calculate final TF-IDF scores and document norms concurrently
	wg.Add(len(termFreq))
	for docID, terms := range termFreq {
		se.workerPool <- struct{}{} // Acquire worker

		go func(docID int, terms map[string]int) {
			defer wg.Done()
			defer func() { <-se.workerPool }() // Release worker

			docScores := make(map[string]float64)
			var norm float64

			docLen := float64(len(terms))
			for term, freq := range terms {
				tfidf := (float64(freq) / docLen) * se.idf[term]
				docScores[term] = tfidf
				norm += tfidf * tfidf
			}

			se.mu.Lock()
			se.tfidf[docID] = docScores
			se.docNorms[docID] = math.Sqrt(norm)
			se.mu.Unlock()

		}(docID, terms)
	}

	wg.Wait()
	fmt.Printf("✓ TF-IDF calculation completed in %v\n", time.Since(start))
}

// Search performs search operation with query
func (se *SearchEngine) Search(query string) []SearchResult {
	start := time.Now()
	fmt.Printf("\n🔍 Searching for: %s\n", query)

	// Preprocess query
	queryTokens := se.preprocessor.preprocess(query)
	queryVector := make(map[string]float64)
	var queryNorm float64

	// Calculate query vector
	for _, token := range queryTokens {
		if _, isStopWord := se.stopWords[token]; !isStopWord {
			queryVector[token]++
		}
	}

	// Normalize query vector
	queryLen := float64(len(queryTokens))
	for term := range queryVector {
		if idf, exists := se.idf[term]; exists {
			queryVector[term] = (queryVector[term] / queryLen) * idf
			queryNorm += queryVector[term] * queryVector[term]
		}
	}
	queryNorm = math.Sqrt(queryNorm)

	// Calculate scores concurrently
	var wg sync.WaitGroup
	scoresChan := make(chan SearchResult, len(se.tfidf))

	for docID := range se.tfidf {
		wg.Add(1)
		se.workerPool <- struct{}{} // Acquire worker

		go func(docID int) {
			defer wg.Done()
			defer func() { <-se.workerPool }() // Release worker

			var score float64
			docVec := se.tfidf[docID]
			docNorm := se.docNorms[docID]

			for term, queryWeight := range queryVector {
				if docWeight, exists := docVec[term]; exists {
					score += (queryWeight * docWeight) / (queryNorm * docNorm)
				}
			}

			if score > 0 {
				scoresChan <- SearchResult{ID: docID, Score: score}
			}
		}(docID)
	}

	// Close channel when all workers are done
	go func() {
		wg.Wait()
		close(scoresChan)
	}()

	// Collect and sort results
	var results []SearchResult
	for result := range scoresChan {
		results = append(results, result)
	}

	// Sort results by score (using quick sort)
	quickSort(results, 0, len(results)-1)

	fmt.Printf("✓ Search completed in %v, found %d results\n", time.Since(start), len(results))
	return results
}

// Quick sort implementation for search results
func quickSort(results []SearchResult, low, high int) {
	if low < high {
		pivot := partition(results, low, high)
		quickSort(results, low, pivot-1)
		quickSort(results, pivot+1, high)
	}
}

func partition(results []SearchResult, low, high int) int {
	pivot := results[high].Score
	i := low - 1

	for j := low; j < high; j++ {
		if results[j].Score > pivot {
			i++
			results[i], results[j] = results[j], results[i]
		}
	}

	results[i+1], results[high] = results[high], results[i+1]
	return i + 1
}