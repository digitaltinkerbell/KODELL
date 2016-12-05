#pragma once

#include "../config/types.h"
#include "../project/network_model.h"

class CKDConfigManager;
class CKDDataSet;
class CKDNetworkModel;
class CKDTrainMode;
class CKDMiniBatch;
class CKDTensor;
class CKDTensor;
class CKDReporterPool;
class CKDLearningEngine;
class CKDDimension;

struct CKDTestValidateInfo;

class CKDProjectManager {
public:
	CKDProjectManager(CKDConfigManager* pConfMan, const char* name);
	virtual ~CKDProjectManager();

	void Execute();

	const char* GetProjectName() const { return m_pProjectName; }
	const char* GetProjectDesc() const { return m_pProjectDesc; }

	double GetLearningRate() const;

	CKDDimension GetInputDimension() const;
	CKDDimension GetOutputDimension() const;

	enumProjectMode GetProjectMode() const { return m_projectMode; }
	enumProjectType GetProjectType() const { return m_projectType; }
	enumTrainMethod GetTrainMethod() const;

	int GetIterateCount() const;

	CKDReporterPool* GetReporterPool() const;
	CKDNetworkModel* GetNetworkModel() const;

	CKDDataSet* GetDataSet() const;

	int GetMinibatchSize() const;
	enum enumMinibatchType GetMinibatchType() const;
	
	int GetTrainExampleCount() const;
	int GetTestExampleCount(bool testset) const;
	
	int GetTestCount(bool testset) const;

	bool GetNthTrainExample(int nth, CKDTensor& input, CKDTensor& output) const;
	bool GetNthTestExample(bool testset, int nth, CKDTensor& input, CKDTensor& output) const;

	void EvalRegularCost(const CKDTensor &estimate, const CKDTensor &answer, CKDLearningEngine* pEngine, double* pCost) const;
	void EvalRegularGrad(const CKDTensor &estimate, const CKDTensor &answer, CKDLearningEngine* pEngine, CKDTensor* pGrad) const;

	double GetWeightDecayNorm2Ratio() const;

	CKDTestValidateInfo* GetValidateSettings();

protected:
	XMLNode* m_projectNode;
	
	CKDProjectManager* m_pBaseProject;

	const char* m_pProjectName;
	const char* m_pDummyMsg;
	
	char* m_pProjectDesc;
	
	CKDDataSet* m_pDataSet;
	CKDNetworkModel* m_pNetworkModel;
	CKDTrainMode* m_pTrainMode;

	enum enumProjectMode m_projectMode;
	enum enumProjectType m_projectType;
};
