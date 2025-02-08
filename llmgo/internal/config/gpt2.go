package config

import "fmt"

type GPT2Config struct {
	MaxSeqLen       int `json:"max_seq_len"`
	VocabSize       int `json:"vocab_size"`
	NumLayers       int `json:"num_layers"`
	NumHeads        int `json:"num_heads"`
	Channels        int `json:"channels"`
	PaddedVocabSize int `json:"padded_vocab_size"`
}

func NewDefaultConfig() *GPT2Config {
	return &GPT2Config{
		MaxSeqLen:       1024,
		VocabSize:       50257,
		PaddedVocabSize: 50304,
		NumLayers:       12,
		NumHeads:        12,
		Channels:        768,
	}
}

func (c *GPT2Config) Validate() error {
	if c.MaxSeqLen <= 0 {
		return fmt.Errorf("max_seq_len must be greater than 0")
	}
	if c.VocabSize <= 0 {
		return fmt.Errorf("vocab_size must be greater than 0")
	}
	if c.PaddedVocabSize <= 0 {
		return fmt.Errorf("padded_vocab_size must be greater than 0")
	}
	if c.NumLayers <= 0 {
		return fmt.Errorf("num_layers must be greater than 0")
	}
	if c.NumHeads <= 0 {
		return fmt.Errorf("num_heads must be greater than 0")
	}
	if c.Channels <= 0 {
		return fmt.Errorf("channels must be greater than 0")
	}
	return nil
}
