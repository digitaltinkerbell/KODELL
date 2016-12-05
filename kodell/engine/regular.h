#pragma once

#include "../config/types.h"

class CKDTensor;
class CKDLearningEngine;

class CKDRegular {
public:
	CKDRegular();
	virtual ~CKDRegular();

	void Setup(XMLNode* node);

	void EvalRegularCost(const CKDTensor &estimate, const CKDTensor &answer, CKDLearningEngine* pEngine, double* pCost) const;
	void EvalRegularGrad(const CKDTensor &estimate, const CKDTensor &answer, CKDLearningEngine* pEngine, CKDTensor* pGrad) const;

	double GetWeightDecayNorm2Ratio() const;

protected:
	enum enumRegularType m_type;
	enum enumRegulatrDest m_dest;
	double m_ratio;
};

class CKDRegularPool {
public:
	CKDRegularPool();
	virtual ~CKDRegularPool();

	void Setup(XMLNode* node);

	void EvalRegularCost(const CKDTensor &estimate, const CKDTensor &answer, CKDLearningEngine* pEngine, double* pCost) const;
	void EvalRegularGrad(const CKDTensor &estimate, const CKDTensor &answer, CKDLearningEngine* pEngine, CKDTensor* pGrad) const;

	double GetWeightDecayNorm2Ratio() const;

protected:
	int m_nRegular;
	CKDRegular* m_pRegulars;
};

