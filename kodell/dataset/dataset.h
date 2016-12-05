#pragma once

#include "../config/types.h"
#include "../math/tensor.h"

class CKDProjectManager;

class CKDDataSet {
public:
	CKDDataSet(XMLNode* node);
	virtual ~CKDDataSet();

	virtual CKDDimension GetInputDimension() const { return m_inputDim; }
	virtual CKDDimension GetOutputDimension() const { return m_outputDim; }

	virtual int GetTrainExampleCount() const { return m_nTrainExampleCount; }
	virtual int GetTestExampleCount() const { return m_nTestExampleCount; }

	virtual bool GetNthTrainExample(int nth, CKDTensor& input, CKDTensor& output) const;
	virtual bool GetNthTestExample(int nth, CKDTensor& input, CKDTensor& output) const;

	void SplitTestSet(XMLNode* split);

	static CKDDataSet* Create(CKDProjectManager* pConfig, XMLNode* node);
	static int ms_compare(const void* p1, const void* p2);

protected:
	CKDDimension m_inputDim;
	CKDDimension m_outputDim;

	int m_nTrainExampleCount;
	int m_nTestExampleCount;

	int m_nExampleCount;

	int* m_pTrainIndexes;
	int* m_pTestIndexes;

	double* m_pExamples;

	//enum enumOutputType m_outputType;

	static XMLNode* ms_seekDataSetClause(XMLNode* node);
};

class CKDClosedDataSet : public CKDDataSet {
public:
	CKDClosedDataSet(XMLNode* node);
	virtual ~CKDClosedDataSet();

protected:
};

class CKDLoadedDataSet : public CKDDataSet {
public:
	CKDLoadedDataSet(CKDProjectManager* pConfig, XMLNode* node, const char* source);
	virtual ~CKDLoadedDataSet();

protected:
};

class CKDMnistDataSet : public CKDDataSet {
public:
	CKDMnistDataSet(XMLNode* node, const char* source);
	virtual ~CKDMnistDataSet();

	bool GetNthTrainExample(int nth, CKDTensor& input, CKDTensor& output) const;
	bool GetNthTestExample(int nth, CKDTensor& input, CKDTensor& output) const;

protected:
	unsigned char* m_pTrainInputPool;
	unsigned char* m_pTrainOutputPool;
	unsigned char* m_pTestInputPool;
	unsigned char* m_pTestOutputPool;

	int m_readIntMSB(FILE* fid);
};

class CKDCifar10DataSet : public CKDDataSet {
public:
	CKDCifar10DataSet(XMLNode* node, const char* source);
	virtual ~CKDCifar10DataSet();

	bool GetNthTrainExample(int nth, CKDTensor& input, CKDTensor& output) const;
	bool GetNthTestExample(int nth, CKDTensor& input, CKDTensor& output) const;

protected:
	unsigned char* m_pTrainPool;
	unsigned char* m_pTestPool;

	void m_rearrange_data_buffer(CKDTensor& input, const unsigned char* pBuf) const;
};

enum reberAlphabet { reber_B, reber_P, reber_S, reber_T, reber_V, reber_X, reber_E, reber_0=0, };

class CKDEmbeddedReberDataSet : public CKDDataSet {
public:
	CKDEmbeddedReberDataSet(XMLNode* node);
	virtual ~CKDEmbeddedReberDataSet();

	bool GetNthTrainExample(int nth, CKDTensor& input, CKDTensor& output) const;
	bool GetNthTestExample(int nth, CKDTensor& input, CKDTensor& output) const;

protected:
	void m_generateSentence(int nth, CKDTensor& input, CKDTensor& output) const;
	int m_move_state(enum reberAlphabet alphabet, int next_state, int *alphabets, int& idx) const;

};

class CKDAccPlusOneDataSet : public CKDDataSet {
public:
	CKDAccPlusOneDataSet(XMLNode* node);
	virtual ~CKDAccPlusOneDataSet();

	bool GetNthTrainExample(int nth, CKDTensor& input, CKDTensor& output) const;
	bool GetNthTestExample(int nth, CKDTensor& input, CKDTensor& output) const;

protected:
	void m_generateSentence(int nth, CKDTensor& input, CKDTensor& output) const;
};

class CKDBinaryFilter {
public:
	CKDBinaryFilter();
	virtual ~CKDBinaryFilter();

	void Setup(XMLNode* filter);

	bool Match(const char* value, int* pResult) const;

protected:
	const char* m_value;
	int m_result;
};

class CKDBinaryFilters {
public:
	CKDBinaryFilters(XMLNode* node);
	virtual ~CKDBinaryFilters();

	bool Match(const char* str, int* pnum) const;

protected:
	int m_nCount;
	CKDBinaryFilter* m_pFilters;
};

class CKDClassifyClass {
public:
	CKDClassifyClass();
	virtual ~CKDClassifyClass();

	void Setup(XMLNode* filter);

	bool Match(const char* value, int* pIndex) const;

protected:
	const char* m_value;
	int m_index;
};

class CKDClassifyClasses {
public:
	CKDClassifyClasses(XMLNode* node);
	virtual ~CKDClassifyClasses();

	bool Match(const char* str, int* pnum) const;

protected:
	int m_nCount;
	CKDClassifyClass* m_pClasses;
};
