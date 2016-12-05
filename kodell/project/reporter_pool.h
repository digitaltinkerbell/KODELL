#pragma once

#include "../project/reporter.h"

class CKDContextReportTestStepInfo;
class CKDContextReportTestFullInfo;

class CKDReporterPool {
public:
	CKDReporterPool();
	virtual ~CKDReporterPool();

	void Setup(XMLNode* node);

	void ReportDebug(const char* caption, const CKDTensor& tensor);

	void ReportTrainStart(const char* pProjectName);
	void ReportTrainExit(const char* pReason);
	void ReportTrainEnd(int nStep);

	void ReportIterateStepStart(int nStep);
	void ReportIterateStepEnd();

	void ReportMinibatchSelect(int nth, int src_idx);
	void ReportMinibatchVomit(int nth);

	void ReportExample(CKDTensor input, CKDTensor output);
	void ReportSetLayer(int nLayer, int nLayerCount, enum enumTrainPhase trainPhase);

	void ReportLayerLinearEval(CKDTensor weight, CKDTensor bias, CKDTensor net);
	void ReportLayerActivateEval(enum actFuncType func, CKDTensor out);

	void ReportCost(double cost, bool reset, bool close);
	void ReportCostGradient(CKDTensor grad);

	void ReportLayerGradient(CKDTensor grad);
	void ReportLayerWeightGradDelta(CKDTensor delta, CKDTensor acc);
	void ReportLayerBiasGradDelta(CKDTensor delta, CKDTensor acc);
	void ReportLayerBiasGradDelta(double delta, double acc);
	void ReportLayerInputGrad(CKDTensor delta_input);

	void ReportLayerFeedSetting(int batch_size, double rate);
	void ReportLayerFeedWeight(CKDTensor acc_delta, CKDTensor weight);
	void ReportLayerFeedBias(CKDTensor acc_delta, CKDTensor bias);
	void ReportLayerFeedBias(double acc_delta, CKDTensor bias);
	
	void ReportFinalResult(CKDTensor weight, CKDTensor bias);
	void ReportFinalResult(CKDTensor weight1, CKDTensor weight2, CKDTensor bias);

	void ReportTestStart(bool test);
	//void ReportTestCase(bool test, int nth, CKDTensor input, CKDTensor output, CKDTensor output_est, bool is_correct);
	//void ReportTestResult(bool test, int nTestCount, int nCorrectCount);
	void ReportTestCase(CKDReportInfo& info);
	void ReportTestResult(CKDReportInfo& info);

protected:
	int m_nReporterCount;
	int m_nIterateStep;
	int m_nLayer;

	unsigned int m_nReportFlags;

	CKDReporter* m_pReporters;
	
	bool m_flagCheck(int flag) const { return this && ((m_nReportFlags & flag) != 0); }
};