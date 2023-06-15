package openai

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
)

// Chat message role defined by the OpenAI API.
const (
	ChatMessageRoleSystem    = "system"
	ChatMessageRoleUser      = "user"
	ChatMessageRoleAssistant = "assistant"
	ChatMessageRoleFunction  = "function"
)

var (
	ErrChatCompletionInvalidModel       = errors.New("this model is not supported with this method, please use CreateCompletion client method instead") //nolint:lll
	ErrChatCompletionStreamNotSupported = errors.New("streaming is not supported with this method, please use CreateChatCompletionStream")              //nolint:lll
	ErrModelNotSupportedWithPlugins     = errors.New("this model is not supported with plugins")                                                        //nolint:lll
)

type Arguments string

func (a Arguments) Decode(v any) error {
	return json.Unmarshal([]byte(a), v)
}
func (a Arguments) String() string {
	return string(a)
}

type FunctionCall struct {
	Name      string    `json:"name,omitempty"`
	Arguments Arguments `json:"arguments,omitempty"`
}

var zeroFunctionCall = FunctionCall{}

type ChatCompletionMessage struct {
	Role         string       `json:"role"`
	Content      string       `json:"content"`
	FunctionCall FunctionCall `json:"function_call,omitempty"`

	// This property isn't in the official documentation, but it's in
	// the documentation for the official library for python:
	// - https://github.com/openai/openai-python/blob/main/chatml.md
	// - https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
	Name string `json:"name,omitempty"`
}

func (c ChatCompletionMessage) MarshalJSON() ([]byte, error) {
	// We need to use a custom marshaler because the FunctionCall field
	// is a pointer, and we want to omit it if it's nil.
	type Alias ChatCompletionMessage
	if c.FunctionCall == zeroFunctionCall {
		return json.Marshal(&struct {
			FunctionCall *FunctionCall `json:"function_call,omitempty"`
			Alias
		}{
			FunctionCall: nil,
			Alias:        (Alias)(c),
		})
	}
	return json.Marshal(&struct {
		*Alias
		FunctionCall *FunctionCall `json:"function_call,omitempty"`
	}{
		Alias:        (*Alias)(&c),
		FunctionCall: &c.FunctionCall,
	})
}

type JSONSchemaType string

const (
	JSONSchemaTypeObject  JSONSchemaType = "object"
	JSONSchemaTypeNumber  JSONSchemaType = "number"
	JSONSchemaTypeString  JSONSchemaType = "string"
	JSONSchemaTypeArray   JSONSchemaType = "array"
	JSONSchemaTypeNull    JSONSchemaType = "null"
	JSONSchemaTypeBoolean JSONSchemaType = "boolean"
)

type JSONSchema struct {
	Type        JSONSchemaType         `json:"type,omitempty"`
	Description string                 `json:"description,omitempty"`
	Enum        []string               `json:"enum,omitempty"`
	Properties  map[string]*JSONSchema `json:"properties,omitempty"`
	Required    []string               `json:"required,omitempty"`
}
type FuncParameters struct {
	Type       JSONSchemaType        `json:"type"`
	Properties map[string]JSONSchema `json:"properties"`
	Required   []string              `json:"required,omitempty"`
}
type Functions struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  FuncParameters `json:"parameters"`
}

// ChatCompletionRequest represents a request structure for chat completion API.
type ChatCompletionRequest struct {
	Model            string                  `json:"model"`
	Messages         []ChatCompletionMessage `json:"messages"`
	MaxTokens        int                     `json:"max_tokens,omitempty"`
	Temperature      float32                 `json:"temperature,omitempty"`
	TopP             float32                 `json:"top_p,omitempty"`
	N                int                     `json:"n,omitempty"`
	Stream           bool                    `json:"stream,omitempty"`
	Stop             []string                `json:"stop,omitempty"`
	PresencePenalty  float32                 `json:"presence_penalty,omitempty"`
	FrequencyPenalty float32                 `json:"frequency_penalty,omitempty"`
	LogitBias        map[string]int          `json:"logit_bias,omitempty"`
	User             string                  `json:"user,omitempty"`
	Functions        []Functions             `json:"functions,omitempty"`
}

type FinishReason string

const (
	FinishReasonStop          FinishReason = "stop"
	FinishReasonLength        FinishReason = "length"
	FinishReasonFunctionCall  FinishReason = "function_call"
	FinishReasonContentFilter FinishReason = "content_filter"
	FinishReasonNull          FinishReason = "null"
)

type ChatCompletionChoice struct {
	Index   int                   `json:"index"`
	Message ChatCompletionMessage `json:"message"`
	// FinishReason
	// stop: API returned complete message,
	// or a message terminated by one of the stop sequences provided via the stop parameter
	// length: Incomplete model output due to max_tokens parameter or token limit
	// function_call: The model decided to call a function
	// content_filter: Omitted content due to a flag from our content filters
	// null: API response still in progress or incomplete
	FinishReason FinishReason `json:"finish_reason"`
}

// ChatCompletionResponse represents a response structure for chat completion API.
type ChatCompletionResponse struct {
	ID      string                 `json:"id"`
	Object  string                 `json:"object"`
	Created int64                  `json:"created"`
	Model   string                 `json:"model"`
	Choices []ChatCompletionChoice `json:"choices"`
	Usage   Usage                  `json:"usage"`
}

// CreateChatCompletion — API call to Create a completion for the chat message.
func (c *Client) CreateChatCompletion(
	ctx context.Context,
	request ChatCompletionRequest,
) (response ChatCompletionResponse, err error) {
	if request.Stream {
		err = ErrChatCompletionStreamNotSupported
		return
	}

	if !checkModelSupportsPlugins(request.Model) {
		err = ErrModelNotSupportedWithPlugins
		return
	}

	urlSuffix := "/chat/completions"
	if !checkEndpointSupportsModel(urlSuffix, request.Model) {
		err = ErrChatCompletionInvalidModel
		return
	}

	req, err := c.requestBuilder.Build(ctx, http.MethodPost, c.fullURL(urlSuffix, request.Model), request)
	if err != nil {
		return
	}

	err = c.sendRequest(req, &response)
	return
}
