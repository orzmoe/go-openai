package openai

import (
	"bufio"
	"context"
	"encoding/json"

	utils "github.com/sashabaranov/go-openai/internal"
)

type ChatCompletionStreamChoiceDelta struct {
	Content      string       `json:"content,omitempty"`
	Role         string       `json:"role,omitempty"`
	FunctionCall FunctionCall `json:"function_call,omitempty"`
}

func (c ChatCompletionStreamChoiceDelta) MarshalJSON() ([]byte, error) {
	// We need to use a custom marshaler because the FunctionCall field
	// is a pointer, and we want to omit it if it's nil.
	type Alias ChatCompletionStreamChoiceDelta
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

type ChatCompletionStreamChoice struct {
	Index int                             `json:"index"`
	Delta ChatCompletionStreamChoiceDelta `json:"delta"`
	// FinishReason
	// stop: API returned complete message,
	// or a message terminated by one of the stop sequences provided via the stop parameter
	// length: Incomplete model output due to max_tokens parameter or token limit
	// function_call: The model decided to call a function
	// content_filter: Omitted content due to a flag from our content filters
	// null: API response still in progress or incomplete
	FinishReason FinishReason `json:"finish_reason"`
}

type ChatCompletionStreamResponse struct {
	ID      string                       `json:"id"`
	Object  string                       `json:"object"`
	Created int64                        `json:"created"`
	Model   string                       `json:"model"`
	Choices []ChatCompletionStreamChoice `json:"choices"`
}

// ChatCompletionStream
// Note: Perhaps it is more elegant to abstract Stream using generics.
type ChatCompletionStream struct {
	*streamReader[ChatCompletionStreamResponse]
}

// CreateChatCompletionStream — API call to create a chat completion w/ streaming
// support. It sets whether to stream back partial progress. If set, tokens will be
// sent as data-only server-sent events as they become available, with the
// stream terminated by a data: [DONE] message.
func (c *Client) CreateChatCompletionStream(
	ctx context.Context,
	request ChatCompletionRequest,
) (stream *ChatCompletionStream, err error) {
	if !checkModelSupportsPlugins(request.Model) {
		err = ErrModelNotSupportedWithPlugins
		return
	}

	urlSuffix := "/chat/completions"
	if !checkEndpointSupportsModel(urlSuffix, request.Model) {
		err = ErrChatCompletionInvalidModel
		return
	}

	request.Stream = true
	req, err := c.newStreamRequest(ctx, "POST", urlSuffix, request, request.Model)
	if err != nil {
		return
	}

	resp, err := c.config.HTTPClient.Do(req) //nolint:bodyclose // body is closed in stream.Close()
	if err != nil {
		return
	}
	if isFailureStatusCode(resp) {
		return nil, c.handleErrorResp(resp)
	}

	stream = &ChatCompletionStream{
		streamReader: &streamReader[ChatCompletionStreamResponse]{
			emptyMessagesLimit: c.config.EmptyMessagesLimit,
			reader:             bufio.NewReader(resp.Body),
			response:           resp,
			errAccumulator:     utils.NewErrorAccumulator(),
			unmarshaler:        &utils.JSONUnmarshaler{},
		},
	}
	return
}
