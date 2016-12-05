#pragma once

#include "../config/types.h"
#include "../util/xml_util.h"

#define REPORT_FLAG_FULL	0xFFFF
#define REPORT_FLAG_BRIEF	0x000F;

class CKDProjectManager;
class CKDReporterPool;
class CKDTestConfig;
class CKDTensor;
class CKDTensor;
class CKDRegularPool;
class CKDLearningEngine;

struct CKDTestValidateInfo;

class CKDTrainMode {
public:
	CKDTrainMode(XMLNode* node, CKDProjectManager* pManager);
	virtual ~CKDTrainMode();

	enumTrainMethod GetTrainMethod() const { return m_method;  }

	double GetLearningRate() const { return m_rate; }
	int GetIterateCount() const { return m_iterate; }
	//int GetReportFlags(int nStep) const;

	int GetMinibatchSize() const { return m_minibatchSize; }
	enumMinibatchType GetMinibatchType() const { return m_minitachType; }

	CKDReporterPool* GetReporterPool() const { return m_pReporterPool; }

	//int GetTestExampleCount() const;

	int GetTestCount(bool testset) const;

	//bool GetNthTestExample(int nth, CKDTensor& input, CKDTensor& output) const;

	void EvalRegularCost(const CKDTensor &estimate, const CKDTensor &answer, CKDLearningEngine* pEngine, double* pCost) const;
	void EvalRegularGrad(const CKDTensor &estimate, const CKDTensor &answer, CKDLearningEngine* pEngine, CKDTensor* pGrad) const;

	double GetWeightDecayNorm2Ratio() const;

	CKDTestValidateInfo* GetValidateSettings();

protected:
	CKDProjectManager* m_pManager;

	enumTrainMethod m_method;
	enumMinibatchType m_minitachType;

	int m_iterate;
	int m_minibatchSize;
	
	//int m_reportCount;
	//CKDReportSetting* m_reportSettings;
	CKDRegularPool* m_pRegularPool;
	CKDReporterPool* m_pReporterPool;
	CKDTestConfig* m_pTestConfig;

	double m_rate;
};