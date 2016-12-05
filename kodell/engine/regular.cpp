#include "stdafx.h"

#include "../engine/regular.h"
#include "../engine/learning_engine.h"
#include "../math/tensor.h"
#include "../util/util.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

CKDRegular::CKDRegular() {
	m_type = regular_type_undef;
	m_dest = regular_dest_undef;
	m_ratio = 0;
}

CKDRegular::~CKDRegular() {
}

void CKDRegular::Setup(XMLNode* node) {
	static const char* type_names[] = { "norm1_decay", "norm2_decay", NULL };
	static const char* dest_names[] = { "all", NULL };

	m_type = (enum enumRegularType) CKDXmlUtil::GetEnumAttr(node, "type", type_names);
	m_dest = (enum enumRegulatrDest) CKDXmlUtil::GetEnumAttr(node, "dest", dest_names);

	m_ratio = CKDXmlUtil::GetDoubleAttr(node, "ratio");
}

void CKDRegular::EvalRegularCost(const CKDTensor &estimate, const CKDTensor &answer, CKDLearningEngine* pEngine, double* pCost) const {
	if (m_dest != regular_dest_all) THROW_TEMP;

	if (m_type == regular_type_norm2) {
		*pCost += m_ratio * pEngine->GetWeightSqSum();
	}
	else if (m_type == regular_type_norm1) {
		*pCost += m_ratio * pEngine->GetWeightAbsSum();
	}
	else {
		THROW_TEMP;
	}
}

void CKDRegular::EvalRegularGrad(const CKDTensor &estimate, const CKDTensor &answer, CKDLearningEngine* pEngine, CKDTensor* pGrad) const {
}

double CKDRegular::GetWeightDecayNorm2Ratio() const {
	if (m_type != regular_type_norm2 || m_type != regular_type_norm1) return 0;
	if (m_dest != regular_dest_all) return 0;

	return m_ratio;
}

CKDRegularPool::CKDRegularPool() {
	m_nRegular = 0;
	m_pRegulars = NULL;
}

CKDRegularPool::~CKDRegularPool() {
	delete[] m_pRegulars;
}


void CKDRegularPool::Setup(XMLNode* node) {
	int nCount = CKDXmlUtil::GetTagChildCount(node, "regular");

	XMLNode* child = CKDXmlUtil::SeekTagChild(node, "regular");

	m_nRegular = 0;
	m_pRegulars = new CKDRegular[nCount];

	for (int i = 0; i < nCount; i++) {
		m_pRegulars[m_nRegular].Setup(child);
		m_nRegular++;

		child = CKDXmlUtil::SeekTagSibling(child, "regular");
	}
}

void CKDRegularPool::EvalRegularCost(const CKDTensor &estimate, const CKDTensor &answer, CKDLearningEngine* pEngine, double* pCost) const {
	for (int i = 0; i < m_nRegular; i++) {
		m_pRegulars[i].EvalRegularCost(estimate, answer, pEngine, pCost);
	}
}

void CKDRegularPool::EvalRegularGrad(const CKDTensor &estimate, const CKDTensor &answer, CKDLearningEngine* pEngine, CKDTensor* pGrad) const {
	for (int i = 0; i < m_nRegular; i++) {
		m_pRegulars[i].EvalRegularGrad(estimate, answer, pEngine, pGrad);
	}
}

double CKDRegularPool::GetWeightDecayNorm2Ratio() const {
	for (int i = 0; i < m_nRegular; i++) {
		double ratio = m_pRegulars[i].GetWeightDecayNorm2Ratio();
		if (ratio) return ratio;
	}

	return 0;
}