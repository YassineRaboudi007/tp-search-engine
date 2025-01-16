package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
)

// CommandRequest represents the JSON input structure
type CommandRequest struct {
	Command   string   `json:"command"`
	Path      string   `json:"path,omitempty"`
	Query     string   `json:"query,omitempty"`
	Documents []string `json:"documents,omitempty"`
}

// SearchResponse represents the JSON output structure
type SearchResponse struct {
	Results []SearchResult `json:"results,omitempty"`
	Error   string        `json:"error,omitempty"`
	Status  string        `json:"status"`
}

func main() {
	// Define flags for different modes of operation
	inputFile := flag.String("input", "", "JSON input file path")
	outputFile := flag.String("output", "", "JSON output file path")
	flag.Parse()

	if *inputFile == "" {
		log.Fatal("Input file path is required")
	}

	// Read input JSON
	inputData, err := os.ReadFile(*inputFile)
	if err != nil {
		writeError(fmt.Sprintf("Error reading input file: %v", err), *outputFile)
		return
	}

	var request CommandRequest
	if err := json.Unmarshal(inputData, &request); err != nil {
		writeError(fmt.Sprintf("Error parsing input JSON: %v", err), *outputFile)
		return
	}

	// Create search engine instance
	engine := NewSearchEngine(4) // Use 4 workers

	var response SearchResponse

	switch request.Command {
	case "index":
		if request.Path != "" {
			// Index documents from directory
			err = engine.LoadDocuments(request.Path)
		} else if len(request.Documents) > 0 {
			// Index provided documents
			err = engine.IndexDocuments(request.Documents)
		} else {
			err = fmt.Errorf("either path or documents must be provided for indexing")
		}

		if err != nil {
			response.Error = err.Error()
			response.Status = "error"
		} else {
			engine.CalculateTFIDF()
			response.Status = "success"
		}

	case "search":
		if request.Query == "" {
			response.Error = "query is required for search"
			response.Status = "error"
		} else {
			results := engine.Search(request.Query)
			response.Results = results
			response.Status = "success"
		}

	default:
		response.Error = "invalid command"
		response.Status = "error"
	}

	// Write response
	writeResponse(response, *outputFile)
}

func writeError(errMsg, outputFile string) {
	response := SearchResponse{
		Error:  errMsg,
		Status: "error",
	}
	writeResponse(response, outputFile)
}

func writeResponse(response SearchResponse, outputFile string) {
	responseJSON, err := json.MarshalIndent(response, "", "  ")
	if err != nil {
		log.Fatalf("Error creating response JSON: %v", err)
	}

	if outputFile == "" {
		// Write to stdout if no output file specified
		fmt.Println(string(responseJSON))
		return
	}

	if err := os.WriteFile(outputFile, responseJSON, 0644); err != nil {
		log.Fatalf("Error writing response: %v", err)
	}
}

// Add method to SearchEngine to index provided documents
func (se *SearchEngine) IndexDocuments(documents []string) error {
	fmt.Printf("\n📚 Indexing %d documents...\n", len(documents))
	
	for i, content := range documents {
		doc := Document{
			ID:     i,
			Tokens: se.preprocessor.preprocess(content),
		}
		se.documents = append(se.documents, doc)
		
		if i%100 == 0 {
			fmt.Printf("📝 Processed document %d/%d\n", i+1, len(documents))
		}
	}
	
	return nil
}