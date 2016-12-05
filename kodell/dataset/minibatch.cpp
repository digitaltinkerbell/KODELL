#include "stdafx.h"

#include "stdlib.h"

#include "../dataset/minibatch.h"
#include "../dataset/dataset.h"
#include "../project/project_manager.h"
#include "../project/train_mode.h"
#include "../project/reporter_pool.h"
#include "../math/tensor.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

CKDMinibatch::CKDMinibatch(CKDProjectManager* pConfig, CKDMinibatchContext* pContext, CKDReporterPool* pReporters) {
	m_pContext = pContext;
	m_pReporters = pReporters;

	m_nExampleCount = 0;

	m_inputs = NULL;
	m_outputs = NULL;

	int nDataSetSiZe = pConfig->GetTrainExampleCount();
	int nMinibatchSiZe = pConfig->GetMinibatchSize();

	enum enumMinibatchType type = pConfig->GetMinibatchType();
	if (type == minibatch_undef) type = minibatch_all;
	
	int nSeqBase = 0;

	if (type == minibatch_all) {
		nMinibatchSiZe = nDataSetSiZe;
	}
	else if (type == minibatch_unique) {
		if (nMinibatchSiZe > nDataSetSiZe) nMinibatchSiZe = nDataSetSiZe;
	}
	else if (type == minibatch_sequential) {
		nSeqBase = pContext->GetLastFetchIndex();
	}

	m_inputs = new CKDTensor[nMinibatchSiZe];
	m_outputs = new CKDTensor[nMinibatchSiZe];

	int *pFetchIndex = (type == minibatch_unique) ? new int[nMinibatchSiZe] : NULL;

	for (int i = 0; i < nMinibatchSiZe; i++) {
		if (type == minibatch_all) m_fetch(pConfig, i, i);
		else if (type == minibatch_sequential) m_fetch(pConfig, i, i + nSeqBase);
		else m_fetch(pConfig, i, rand() % nDataSetSiZe, pFetchIndex);
		
		if (type == minibatch_unique) {
			if (m_duplicated(i, pFetchIndex)) {
				m_pReporters->ReportMinibatchVomit(i);
				i--;
				continue;
			}
			
		}

		m_nExampleCount++;
	}

	if (type == minibatch_sequential) {
		pContext->SetLastFetchIndex((nSeqBase+m_nExampleCount)%nDataSetSiZe);
	}
	else if (type == minibatch_unique) {
		delete[] pFetchIndex;
	}
}

CKDMinibatch::~CKDMinibatch() {
	delete [] m_inputs;
	delete [] m_outputs;

}

void CKDMinibatch::m_fetch(const CKDProjectManager* pConfig, int nDestIdx, int nSrcIdx, int* pFetchIndex) {
	pConfig->GetNthTrainExample(nSrcIdx, m_inputs[nDestIdx], m_outputs[nDestIdx]);
	if (pFetchIndex) pFetchIndex[nDestIdx] = nSrcIdx;

	m_pReporters->ReportMinibatchSelect(nDestIdx, nSrcIdx);
}

bool CKDMinibatch::m_duplicated(int nDestIdx, int* pFetchIndex) const {
	int nNomIdx = pFetchIndex[nDestIdx - 1];

	for (int i = 0; i < nDestIdx - 1; i++) {
		if (nNomIdx == pFetchIndex[i]) return true;
	}

	return false;
}

void CKDMinibatch::GetExample(int nth, CKDTensor* pInput, CKDTensor* pOutput) const {
	if (nth < 0 || nth >= m_nExampleCount) THROW_TEMP;
	*pInput = m_inputs[nth];
	*pOutput = m_outputs[nth];
}
