// PicoClaw - Ultra-lightweight personal AI agent
// License: MIT
//
// Copyright (c) 2026 PicoClaw contributors

package providers

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/sipeed/picoclaw/pkg/providers/common"
)



// OllamaCloudProvider implements LLMProvider for Ollama Cloud API
// Uses native Ollama API format (not OpenAI-compatible)
type OllamaCloudProvider struct {
	apiKey     string
	apiBase    string
	httpClient *http.Client
}

// NewOllamaCloudProvider creates a new Ollama Cloud provider
func NewOllamaCloudProvider(apiKey, apiBase, proxy string) *OllamaCloudProvider {
	if apiBase == "" {
		apiBase = "https://ollama.com/api"
	}
	return &OllamaCloudProvider{
		apiKey:     apiKey,
		apiBase:    strings.TrimRight(apiBase, "/"),
		httpClient: common.NewHTTPClient(proxy),
	}
}

// Chat implements providers.LLMProvider
func (p *OllamaCloudProvider) Chat(
	ctx context.Context,
	messages []Message,
	tools []ToolDefinition,
	model string,
	options map[string]any,
) (*LLMResponse, error) {
	if p.apiBase == "" {
		return nil, fmt.Errorf("API base not configured")
	}

	requestBody := p.buildRequestBody(messages, tools, model, options, false)

	jsonData, err := json.Marshal(requestBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", p.apiBase+"/chat", bytes.NewReader(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	if p.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+p.apiKey)
	}

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, common.HandleErrorResponse(resp, p.apiBase)
	}

	return p.parseResponse(resp.Body)
}

// ChatStream implements providers.StreamingProvider
func (p *OllamaCloudProvider) ChatStream(
	ctx context.Context,
	messages []Message,
	tools []ToolDefinition,
	model string,
	options map[string]any,
	onChunk func(accumulated string),
) (*LLMResponse, error) {
	if p.apiBase == "" {
		return nil, fmt.Errorf("API base not configured")
	}

	requestBody := p.buildRequestBody(messages, tools, model, options, true)

	jsonData, err := json.Marshal(requestBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", p.apiBase+"/chat", bytes.NewReader(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	if p.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+p.apiKey)
	}

	// Use a client without Timeout for streaming
	streamClient := &http.Client{Transport: p.httpClient.Transport}
	resp, err := streamClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, common.HandleErrorResponse(resp, p.apiBase)
	}

	return p.parseStreamResponse(ctx, resp.Body, onChunk)
}

// GetDefaultModel returns the default model for this provider
func (p *OllamaCloudProvider) GetDefaultModel() string {
	return "gpt-oss:120b-cloud"
}

// buildRequestBody constructs the Ollama Cloud request body
func (p *OllamaCloudProvider) buildRequestBody(
	messages []Message,
	tools []ToolDefinition,
	model string,
	options map[string]any,
	stream bool,
) map[string]any {
	// Remove protocol prefix if present
	if _, after, found := strings.Cut(model, "/"); found {
		model = after
	}

	requestBody := map[string]any{
		"model":    model,
		"messages": p.serializeMessages(messages),
		"stream":   stream,
	}

	// Add tools if present
	if len(tools) > 0 {
		requestBody["tools"] = p.buildToolsList(tools)
	}

	// Add options
	ollamaOptions := make(map[string]any)

	if maxTokens, ok := common.AsInt(options["max_tokens"]); ok {
		ollamaOptions["num_predict"] = maxTokens
	}

	if temperature, ok := common.AsFloat(options["temperature"]); ok {
		ollamaOptions["temperature"] = temperature
	}

	if len(ollamaOptions) > 0 {
		requestBody["options"] = ollamaOptions
	}

	return requestBody
}

// serializeMessages converts messages to Ollama format
func (p *OllamaCloudProvider) serializeMessages(messages []Message) []map[string]any {
	result := make([]map[string]any, 0, len(messages))
	for _, msg := range messages {
		serialized := map[string]any{
			"role":    msg.Role,
			"content": msg.Content,
		}
		result = append(result, serialized)
	}
	return result
}

// buildToolsList converts tool definitions to Ollama format
func (p *OllamaCloudProvider) buildToolsList(tools []ToolDefinition) []any {
	result := make([]any, 0, len(tools))
	for _, t := range tools {
		result = append(result, t)
	}
	return result
}

// parseResponse parses a non-streaming Ollama Cloud response
func (p *OllamaCloudProvider) parseResponse(reader io.Reader) (*LLMResponse, error) {
	var response struct {
		Model      string `json:"model"`
		CreatedAt  string `json:"created_at"`
		Message    struct {
			Role      string `json:"role"`
			Content   string `json:"content"`
			ToolCalls []struct {
				Function struct {
					Name      string          `json:"name"`
					Arguments json.RawMessage `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"message"`
		DoneReason string `json:"done_reason"`
		Done       bool   `json:"done"`
		EvalCount  int    `json:"eval_count"`
		PromptEvalCount int `json:"prompt_eval_count"`
	}

	if err := json.NewDecoder(reader).Decode(&response); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	llmResp := &LLMResponse{
		Content:      response.Message.Content,
		FinishReason: response.DoneReason,
		Usage: &UsageInfo{
			CompletionTokens: response.EvalCount,
			PromptTokens:     response.PromptEvalCount,
			TotalTokens:      response.EvalCount + response.PromptEvalCount,
		},
	}

	// Parse tool calls if present
	if len(response.Message.ToolCalls) > 0 {
		toolCalls := make([]ToolCall, 0, len(response.Message.ToolCalls))
		for _, tc := range response.Message.ToolCalls {
			args := make(map[string]any)
			if len(tc.Function.Arguments) > 0 {
				json.Unmarshal(tc.Function.Arguments, &args)
			}
			toolCalls = append(toolCalls, ToolCall{
				Name:      tc.Function.Name,
				Arguments: args,
			})
		}
		llmResp.ToolCalls = toolCalls
	}

	return llmResp, nil
}

// parseStreamResponse parses a streaming Ollama Cloud response
func (p *OllamaCloudProvider) parseStreamResponse(
	ctx context.Context,
	reader io.Reader,
	onChunk func(accumulated string),
) (*LLMResponse, error) {
	var textContent strings.Builder
	var finishReason string
	var usage *UsageInfo

	// Tool call assembly
	type toolAccum struct {
		name     string
		argsJSON strings.Builder
	}
	activeTools := map[int]*toolAccum{}

	scanner := bufio.NewScanner(reader)
	for scanner.Scan() {
		// Check for context cancellation
		if err := ctx.Err(); err != nil {
			return nil, err
		}

		line := scanner.Text()
		if line == "" {
			continue
		}

		var chunk struct {
			Model     string `json:"model"`
			CreatedAt string `json:"created_at"`
			Message   struct {
				Role      string `json:"role"`
				Content   string `json:"content"`
				ToolCalls []struct {
					Function struct {
						Name      string          `json:"name"`
						Arguments json.RawMessage `json:"arguments"`
					} `json:"function"`
				} `json:"tool_calls"`
			} `json:"message"`
			Done       bool   `json:"done"`
			DoneReason string `json:"done_reason"`
			EvalCount  int    `json:"eval_count"`
			PromptEvalCount int `json:"prompt_eval_count"`
		}

		if err := json.Unmarshal([]byte(line), &chunk); err != nil {
			continue // skip malformed chunks
		}

		// Accumulate text content
		if chunk.Message.Content != "" {
			textContent.WriteString(chunk.Message.Content)
			if onChunk != nil {
				onChunk(textContent.String())
			}
		}

		// Accumulate tool call deltas
		for i, tc := range chunk.Message.ToolCalls {
			acc, ok := activeTools[i]
			if !ok {
				acc = &toolAccum{}
				activeTools[i] = acc
			}
			if tc.Function.Name != "" {
				acc.name = tc.Function.Name
			}
			if len(tc.Function.Arguments) > 0 {
				acc.argsJSON.Write(tc.Function.Arguments)
			}
		}

		// Check if done
		if chunk.Done {
			finishReason = chunk.DoneReason
			if chunk.EvalCount > 0 || chunk.PromptEvalCount > 0 {
				usage = &UsageInfo{
					CompletionTokens: chunk.EvalCount,
					PromptTokens:     chunk.PromptEvalCount,
					TotalTokens:      chunk.EvalCount + chunk.PromptEvalCount,
				}
			}
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("streaming read error: %w", err)
	}

	// Assemble tool calls from accumulated deltas
	var toolCalls []ToolCall
	for i := 0; i < len(activeTools); i++ {
		acc, ok := activeTools[i]
		if !ok {
			continue
		}
		args := make(map[string]any)
		raw := acc.argsJSON.String()
		if raw != "" {
			if err := json.Unmarshal([]byte(raw), &args); err != nil {
				args["raw"] = raw
			}
		}
		toolCalls = append(toolCalls, ToolCall{
			Name:      acc.name,
			Arguments: args,
		})
	}

	if finishReason == "" {
		finishReason = "stop"
	}

	return &LLMResponse{
		Content:      textContent.String(),
		ToolCalls:    toolCalls,
		FinishReason: finishReason,
		Usage:        usage,
	}, nil
}

// SupportsNativeSearch returns false - Ollama Cloud doesn't support native search
func (p *OllamaCloudProvider) SupportsNativeSearch() bool {
	return false
}
