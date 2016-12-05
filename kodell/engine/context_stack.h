#pragma once

#include "../config/types.h"
#include "../math/tensor.h"

class CKDLayer;
class CKDLearningEngine;
class CKDProjectManager;
class CKDContextReportTestStepInfo;
class CKDContextReportTestFullInfo;

class CKDReportInfo;

class CKDTensorMapNode {
protected:
	friend class CKDLayerInfo;
	friend class CKDContextStack;

	CKDTensorMapNode(const char* pName, int index, CKDTensor tensor);
	virtual ~CKDTensorMapNode();

protected:
	const char* m_pName;
	int m_index;
	CKDTensor m_tensor;

	CKDTensorMapNode* m_pPrevNode;
	CKDTensorMapNode* m_pNextNode;

protected:
	CKDTensorMapNode* seekNode(const char* pName, int index);
	CKDTensorMapNode* seekReverseNode(const char* pName, int index);
};

class CKDLayerInfo {
protected:
	friend class CKDContextStack;

	CKDLayerInfo(CKDLayer* pLayer);
	virtual ~CKDLayerInfo();

protected:
	CKDLayer* m_pLayer;

	CKDLayerInfo* m_pPrevNode;
	CKDLayerInfo* m_pNextNode;

	CKDTensorMapNode* m_pMapHead;
	CKDTensorMapNode* m_pMapTail;

protected:
	CKDTensor getTensor(const char* pName, int index);
	//CKDTensor getRecurrentTensorOnTop(const char* pName);
	void setTensor(const char* pName, CKDTensor tensor, int index);
	//void pushTensor(const char* pName, CKDTensor tensor);
	CKDLayerInfo* seekLayerNode(CKDLayer* pLayer);
};

class CKDContextStack {
public:
	CKDContextStack(CKDLearningEngine* pEngine, enum enumProjectType type, enum enumProjectMode mode, enum enumTestDest dest, int iterate);
	~CKDContextStack();

	void setExample(CKDTensor input, CKDTensor output, int nth);

	bool get_reccurent_next_step();
	bool get_reccurent_prev_step();
	bool need_report_reset();
	bool need_report_close();

	double eval_cost();

	CKDTensor getStepInput();
	CKDTensor getStepOutput();

	void pushLayer(CKDLayer* pLayer);
	void setLayer(CKDLayer* pLayer);

	CKDTensor eval_cost_grad();

	void gradeTestResult(CKDReportInfo& info);
	void getFullGradeResult(CKDReportInfo& info);

	bool get_need_back();
	bool get_need_input_grad();
	bool get_need_recur_grad();

	CKDTensor getInput(int nth = 0);
	CKDTensor getOutput(int nth = 0);
	CKDTensor getGradient(int nth = 0);
	CKDTensor getLayerTensorXXX(const char* pKey);
	CKDTensor getInstantTensor(const char* pKey, int nth = 0);

	double getLayerScalarXXX(const char* pKey);
	double getInstantScalar(const char* pKey, int nth = 0);

	void setOutput(CKDTensor result, int nth = 0);
	void setGradient(CKDTensor gradient, int nth = 0);
	void setCostGradient(CKDTensor gradient, int nth = 0);

	void setLayerTensorXXX(const char* pKey, CKDTensor tensor);
	void setInstantTensor(const char* pKey, CKDTensor tensor, int nth = 0);
	void accumulateLayerTensorXXX(const char* pKey, CKDTensor tensor);
	void accumulateInstantTensor(const char* pKey, CKDTensor tensor, int nth = 0);

protected:
	CKDLearningEngine* m_pEngine;

	enum enumProjectType m_projectType;
	enum enumProjectMode m_projectMode;
	enum enumTestDest m_testDest;
	
	CKDTensor m_exampleInput;
	CKDTensor m_exampleOutput;
	
	CKDTensor m_currGradient;

	int m_iterate;
	int m_exampleIndex;
	int m_sequenceLength;
	int m_sequenceIndex;

	int m_testCount;
	int m_correctCount;

	CKDLayerInfo* m_layerListHead;
	CKDLayerInfo* m_layerListTail;
	CKDLayerInfo* m_currLayer;

protected:
	void m_resetLayerList();

	double m_eval_pure_cost(const CKDTensor &estimate, const CKDTensor &answer);
	double m_eval_regular_cost(const CKDTensor &estimate, const CKDTensor &answer);

	CKDTensor m_eval_pure_cost_grad(const CKDTensor &estimate, const CKDTensor &answer);
	CKDTensor m_eval_regular_cost_grad(const CKDTensor &estimate, const CKDTensor &answer);
};