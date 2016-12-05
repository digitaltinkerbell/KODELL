#pragma once

class CKDLayer;
class CKDTensor;
class CKDProjectManager;
class CKDReporterPool;
class CKDContextStack;

struct CKDTestValidateInfo;

class CKDLearningEngine {
public:
	CKDLearningEngine(CKDProjectManager* pConfig);
	virtual ~CKDLearningEngine();

	void Setup();
	void Train();
	void Test();

	double EvalRegularCost(const CKDTensor &estimate, const CKDTensor &answer);
	CKDTensor EvalRegularGrad(const CKDTensor &estimate, const CKDTensor &answer);

	double GetWeightSqSum();
	double GetWeightAbsSum();

protected:
	CKDProjectManager* m_pConfig;

	int m_nLayerCount;
	CKDLayer** m_ppLayers;

	//enum costFuncType m_funcType;

	//void m_trainStep(CKDContextStack& context, CKDReporterPool* pReporters);
	//CKDTensor m_execForward(CKDTensor input, CKDTensor output, CKDReporterPool* pReporters, bool reset, bool close);
	void m_execForward_step(CKDContextStack& context, CKDReporterPool* pReporters); // , CKDTensor input, CKDTensor output, CKDReporterPool* pReporters, bool reset, bool close);

	void m_execBackward_step(CKDContextStack& context, CKDReporterPool* pReporters); // , CKDTensor grad, CKDReporterPool* pReporters);
	
	void m_resetBackGradient();
	bool m_feedGradient(CKDContextStack& context, double rate, int batch_size, CKDReporterPool* pReporters);
	void m_finalReport(CKDReporterPool* pReporters);

	/*
	double m_eval_cost(const CKDTensor &estimate, const CKDTensor &answer);
	double m_eval_pure_cost(const CKDTensor &estimate, const CKDTensor &answer);
	double m_eval_regular_cost(const CKDTensor &estimate, const CKDTensor &answer);

	CKDTensor m_eval_cost_grad(const CKDTensor &estimate, const CKDTensor &answer);
	CKDTensor m_eval_pure_cost_grad(const CKDTensor &estimate, const CKDTensor &answer);
	CKDTensor m_eval_regular_cost_grad(const CKDTensor &estimate, const CKDTensor &answer);
	*/

	//void m_gradeTrainExamples(CKDTestValidateInfo* pValidate);
	void m_gradeTestExamples(CKDTestValidateInfo* pValidate, bool testset);

	void m_validate(int step, CKDTestValidateInfo* pValidate, CKDReporterPool* pReporters);

};
