#pragma once

#include "../config/types.h"

class CKDTensor;
class CKDProjectManager;
class CKDReporterPool;

class CKDMinibatchContext {
public:
	CKDMinibatchContext() { m_nFetchIndex = 0; }
	int GetLastFetchIndex() const { return m_nFetchIndex; }
	void SetLastFetchIndex(int nIndex) { m_nFetchIndex = nIndex; }

protected:
	int m_nFetchIndex;
};

class CKDMinibatch {
public:
	CKDMinibatch(CKDProjectManager* pConfig, CKDMinibatchContext* pContext, CKDReporterPool* pReporters);
	~CKDMinibatch();

	int size() const { return m_nExampleCount;  }

	void GetExample(int nth, CKDTensor* pInput, CKDTensor* pOutput) const;

protected:
	CKDMinibatchContext* m_pContext;
	CKDReporterPool* m_pReporters;

	int m_nExampleCount;

	CKDTensor* m_inputs;
	CKDTensor* m_outputs;
	
	void m_fetch(const CKDProjectManager* pConfig, int nDestIdx, int nSrcIdx, int* pFetchIndex=NULL);
	
	bool m_duplicated(int nDestIdx, int* pFetchIndex) const;
};