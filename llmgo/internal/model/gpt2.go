package model

import (
	"encoding/binary"
	"fmt"
	"io"
	"llmgo/internal/config"
	"os"
	"sync"
)

type GPT2 struct {
	Config        *config.GPT2Config
	Parameters    []float32
	ParamOffsets  map[string]int
	ParamSizes    map[string]int
	NumParameters int

	mu        sync.Mutex
	BatchSize int
	SeqLen    int
	MeanLoss  float32
}

func NewGPT2FromCheckpoint(checkPointPath string) (*GPT2, error) {
	file, err := os.Open(checkPointPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open checkpoint file: %w", err)
	}
	defer file.Close()

	header := make([]int32, 256)
	if err := binary.Read(file, binary.LittleEndian, header); err != nil {
		return nil, fmt.Errorf("failed to read header: %w", err)
	}

	if header[0] != 20240326 {
		return nil, fmt.Errorf("invalid magic number in checkpoint: %d", header[0])
	}
	if header[1] != 3 {
		return nil, fmt.Errorf("invalid version in checkpoint: %d, expected 3", header[1])
	}

	cfg := &config.GPT2Config{
		MaxSeqLen:       int(header[2]),
		VocabSize:       int(header[3]),
		NumLayers:       int(header[4]),
		NumHeads:        int(header[5]),
		Channels:        int(header[6]),
		PaddedVocabSize: int(header[7]),
	}

	if err := cfg.Validate(); err != nil {
		return nil, fmt.Errorf("invalid config: %w", err)
	}

	model := &GPT2{
		Config:       cfg,
		Parameters:   []float32{},
		ParamOffsets: make(map[string]int),
		ParamSizes:   make(map[string]int),
	}

	model.calculateParameterSizes()

	model.Parameters = make([]float32, model.NumParameters)

	currentPos, err := file.Seek(0, io.SeekCurrent)
	if err != nil {
		return nil, fmt.Errorf("failed to get file position: %w", err)
	}

	fileInfo, err := file.Stat()
	if err != nil {
		return nil, fmt.Errorf("failed to get file info: %w", err)
	}

	remainingBytes := fileInfo.Size() - currentPos
	expectedBytes := int64(model.NumParameters * 4)

	fmt.Printf("Remaining bytes in file: %d\n", remainingBytes)
	fmt.Printf("Expected bytes to read: %d\n", expectedBytes)

	if err := binary.Read(file, binary.LittleEndian, model.Parameters); err != nil {
		return nil, fmt.Errorf("failed to read parameters: %w", err)
	}

	return model, nil
}

func (m *GPT2) calculateParameterSizes() {

	offset := 0
	// First we do token embeddings
	embSize := m.Config.PaddedVocabSize * m.Config.Channels
	m.ParamOffsets["wte"] = offset
	m.ParamSizes["wte"] = embSize
	offset += embSize

	// Next is position embeddings
	posSize := m.Config.MaxSeqLen * m.Config.Channels
	m.ParamOffsets["wpe"] = offset
	m.ParamSizes["wpe"] = posSize
	offset += posSize

	// We have to iterate through the layers now
	for layer := 0; layer < m.Config.NumLayers; layer++ {
		prefix := fmt.Sprintf("Layer %d", layer)

		// LayerNorm 1
		ln1wSize := m.Config.Channels
		m.ParamOffsets[prefix+"_ln1.weight"] = offset
		m.ParamSizes[prefix+"_ln1.weight"] = ln1wSize
		offset += ln1wSize

		// LayerNorm 1 bias
		ln1bSize := m.Config.Channels
		m.ParamOffsets[prefix+"_ln1.bias"] = offset
		m.ParamSizes[prefix+"_ln1.bias"] = ln1bSize
		offset += ln1bSize

		// Attention heads
		qkvwSize := 3 * m.Config.Channels * m.Config.Channels
		m.ParamOffsets[prefix+".attn.qkv.weight"] = offset
		m.ParamSizes[prefix+".attn.qkv.weight"] = qkvwSize
		offset += qkvwSize

		qkvbSize := 3 * m.Config.Channels
		m.ParamOffsets[prefix+".attn.qkv.bias"] = offset
		m.ParamSizes[prefix+".attn.qkv.bias"] = qkvbSize
		offset += qkvbSize

		// Attention projection weights and bias
		attprojwSize := m.Config.Channels * m.Config.Channels
		m.ParamOffsets[prefix+".attn.proj.weight"] = offset
		m.ParamSizes[prefix+".attn.proj.weight"] = attprojwSize
		offset += attprojwSize

		attprojbSize := m.Config.Channels
		m.ParamOffsets[prefix+".attn.proj.bias"] = offset
		m.ParamSizes[prefix+".attn.proj.bias"] = attprojbSize
		offset += attprojbSize

		// Layer norm 2 weights and bias
		ln2wSize := m.Config.Channels
		m.ParamOffsets[prefix+".ln_2.weight"] = offset
		m.ParamSizes[prefix+".ln_2.weight"] = ln2wSize
		offset += ln2wSize

		ln2bSize := m.Config.Channels
		m.ParamOffsets[prefix+".ln_2.bias"] = offset
		m.ParamSizes[prefix+".ln_2.bias"] = ln2bSize
		offset += ln2bSize

		// MLP FC (expansion) weights and bias
		fcwSize := 4 * m.Config.Channels * m.Config.Channels
		m.ParamOffsets[prefix+".mlp.fc.weight"] = offset
		m.ParamSizes[prefix+".mlp.fc.weight"] = fcwSize
		offset += fcwSize

		fcbSize := 4 * m.Config.Channels
		m.ParamOffsets[prefix+".mlp.fc.bias"] = offset
		m.ParamSizes[prefix+".mlp.fc.bias"] = fcbSize
		offset += fcbSize

		// MLP Projection weights and bias
		projwSize := m.Config.Channels * 4 * m.Config.Channels
		m.ParamOffsets[prefix+".mlp.proj.weight"] = offset
		m.ParamSizes[prefix+".mlp.proj.weight"] = projwSize
		offset += projwSize

		projbSize := m.Config.Channels
		m.ParamOffsets[prefix+".mlp.proj.bias"] = offset
		m.ParamSizes[prefix+".mlp.proj.bias"] = projbSize
		offset += projbSize

	}
	//Final layer norm weights and bias
	lnfwSize := m.Config.Channels
	m.ParamOffsets["ln_f.weight"] = offset
	m.ParamSizes["ln_f.weight"] = lnfwSize
	offset += lnfwSize

	lnfbSize := m.Config.Channels
	m.ParamOffsets["ln_f.bias"] = offset
	m.ParamSizes["ln_f.bias"] = lnfbSize
	offset += lnfbSize

	m.NumParameters = offset

	totalParams := 0
	for _, size := range m.ParamSizes {
		totalParams += size
	}

}

func (m *GPT2) GetParameter(name string) ([]float32, error) {

	offset, got := m.ParamOffsets[name]
	if !got {
		return nil, fmt.Errorf("parameter %s not found", name)
	}

	size := m.ParamSizes[name]

	return m.Parameters[offset : offset+size], nil

}
