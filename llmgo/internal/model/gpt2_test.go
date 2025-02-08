package model_test

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"

	"llmgo/internal/model"

	"github.com/stretchr/testify/assert"
)

func TestGPT2FromCheckpointFake(t *testing.T) {
	tmpFile, err := os.CreateTemp("", "gpt2_test")
	if err != nil {
		t.Fatalf("failed to create temp file: %v", err)
	}
	defer os.Remove(tmpFile.Name())

	header := make([]int32, 256)
	header[0] = 20240326
	header[1] = 3
	header[2] = 1024
	header[3] = 50257
	header[4] = 12
	header[5] = 12
	header[6] = 768
	header[7] = 50304

	if err := binary.Write(tmpFile, binary.LittleEndian, header); err != nil {
		t.Fatalf("failed to write header: %v", err)
	}

	totalParams := 124475904

	parameters := make([]float32, totalParams)
	for i := range parameters {
		parameters[i] = 0.01 * float32(i)
	}

	if err := binary.Write(tmpFile, binary.LittleEndian, parameters); err != nil {
		t.Fatalf("failed to write parameters: %v", err)
	}

	tmpFile.Close()

	gpt2Model, err := model.NewGPT2FromCheckpoint(tmpFile.Name())
	assert.Equal(t, gpt2Model.NumParameters, totalParams)
	assert.Nil(t, err)
}

func TestReadHeader(t *testing.T) {
	checkPointPath := "../../../gpt2_124M.bin"
	absPath, _ := filepath.Abs(checkPointPath)

	file, err := os.Open(absPath)
	if err != nil {
		t.Fatalf("failed to open checkpoint file: %v", err)
	}
	defer file.Close()

	header := make([]int32, 256)
	if err := binary.Read(file, binary.LittleEndian, header); err != nil {
		t.Fatalf("failed to read header: %v", err)
	}

	assert.Equal(t, header[0], int32(20240326))
	assert.Equal(t, header[1], int32(3))
	assert.Equal(t, header[2], int32(1024))
	assert.Equal(t, header[3], int32(50257))
	assert.Equal(t, header[4], int32(12))
	assert.Equal(t, header[5], int32(12))
	assert.Equal(t, header[6], int32(768))
	assert.Equal(t, header[7], int32(50304))
}

func TestReadCheckpointFile(t *testing.T) {

	checkPointPath := "../../../gpt2_124M.bin"
	absPath, _ := filepath.Abs(checkPointPath)

	gpt2Model, err := model.NewGPT2FromCheckpoint(absPath)
	assert.Nil(t, err)
	assert.NotNil(t, gpt2Model)

	assert.Equal(t, gpt2Model.Config.MaxSeqLen, 1024)
	assert.Equal(t, gpt2Model.Config.VocabSize, 50257)
	assert.Equal(t, gpt2Model.Config.NumLayers, 12)
	assert.Equal(t, gpt2Model.Config.Channels, 768)
	assert.Equal(t, gpt2Model.Config.PaddedVocabSize, 50304)

	assert.Equal(t, gpt2Model.NumParameters, 124475904)
	assert.Equal(t, len(gpt2Model.Parameters), gpt2Model.NumParameters)

}
