#pragma once

#include "../config/types.h"
#include "../math/tensor.h"

#define REPORT_FLAG_TRAIN_START			0x00000001
#define REPORT_FLAG_TRAIN_EXIT			0x00000002
#define REPORT_FLAG_TRAIN_END			0x00000004
#define REPORT_FLAG_STEP_START			0x00000008
#define REPORT_FLAG_STEP_END			0x00000010
#define REPORT_FLAG_MINIBATCH_SELECT	0x00000020
#define REPORT_FLAG_MINIBATCH_VOMIT		0x00000040
#define REPORT_FLAG_EXAMPLE				0x00000080
#define REPORT_FLAG_SET_LAYER			0x00000100
#define REPORT_FLAG_LAYER_LINEAR_EVAL	0x00000200
#define REPORT_FLAG_LAYER_ACTIVATE_EVAL	0x00000400
#define REPORT_FLAG_COST				0x00000800
#define REPORT_FLAG_COST_AVG			0x02000000
#define REPORT_FLAG_COST_GRAD			0x00001000
#define REPORT_FLAG_LAYER_GRAD			0x00002000
#define REPORT_FLAG_LAYER_GRAD_DELTA	0x00004000
#define REPORT_FLAG_LAYER_BIAS_DELTA	0x00008000
#define REPORT_FLAG_LAYER_INPUT_GRAD	0x00010000
#define REPORT_FLAG_LAYER_FEED_SETTING	0x00020000
#define REPORT_FLAG_LAYER_FEED_WEIGHT	0x00040000
#define REPORT_FLAG_LAYER_FEED_BIAS		0x00080000
#define REPORT_FLAG_FINAL_RESULT		0x00100000
#define REPORT_FLAG_TEST_START			0x00200000
#define REPORT_FLAG_TEST_CASE			0x00400000
#define REPORT_FLAG_TEST_RESULT			0x00800000
#define REPORT_FLAG_DEBUG				0x01000000

#define REPORT_FLAGS_ALL				0xFFFFFFFF
#define REPORT_FLAGS_BRIEF				0x009C088F

class CKDReportNode {
protected:
	friend class CKDReportInfo;

	CKDReportNode(const char* pName, int value);
	virtual ~CKDReportNode();

protected:
	const char* m_pName;
	int m_value;

	CKDReportNode* m_pNextNode;

protected:
	CKDReportNode* seekNode(const char* pName);
};

class CKDReportInfo {
public:
	CKDReportInfo();
	virtual ~CKDReportInfo();

	void setValue(const char* pName, int value);
	int getValue(const char* pName);

protected:
	CKDReportNode* m_pMapHead;
	CKDReportNode* m_pMapTail;

};

struct CKDReporterRange {
	int m_from;
	int m_to;
	int m_offset;
	int m_period;

	void Setup(XMLNode* node);
	bool m_stepCheck(int step) const { return step % m_period == m_offset && step >= m_from && (m_to < 0 || step <= m_to); }
};

class CKDReporter {
protected:
	friend class CKDReporterPool;

	CKDReporter();
	~CKDReporter();

	void Setup(XMLNode* node);

protected:
	int m_nReportFlags;

	int m_nRangeCount;
	CKDReporterRange* m_pRanges;

	enum enumReportDest m_destType;
	FILE* m_fid;

	bool m_flagCheck(int flag) const { return (m_nReportFlags & flag) != 0; }
	bool m_stepCheck(int step) const;
	int m_getTokenMask(const char* token);

protected:
	void ReportDebug(const char* caption, const CKDTensor& tensor);

	void ReportTrainStart(const char* pProjectName);
	void ReportTrainExit(const char* pReason);
	void ReportTrainEnd(int nStep);

	void ReportIterateStepStart(int nStep);
	void ReportIterateStepEnd(int nStep);

	void ReportMinibatchSelect(int nStep, int nth, int src_idx);
	void ReportMinibatchVomit(int nStep, int nth);

	void ReportExample(int nStep, CKDTensor input, CKDTensor output);
	void ReportSetLayer(int nStep, int nLayer, int nLayerCount, enum enumTrainPhase trainPhase);

	void ReportLayerLinearEval(int nStep, int nLayer, CKDTensor weight, CKDTensor bias, CKDTensor net);
	void ReportLayerActivateEval(int nStep, int nLayer, enum actFuncType func, CKDTensor out);

	void ReportCost(int nStep, double cost, bool reset, bool close);
	void ReportCostGradient(int nStep, CKDTensor grad);

	void ReportLayerGradient(int nStep, int nLayer, CKDTensor grad);
	void ReportLayerWeightGradDelta(int nStep, int nLayer, CKDTensor delta, CKDTensor acc);
	void ReportLayerBiasGradDelta(int nStep, int nLayer, CKDTensor delta, CKDTensor acc);
	void ReportLayerBiasGradDelta(int nStep, int nLayer, double delta, double acc);
	void ReportLayerInputGrad(int nStep, int nLayer, CKDTensor delta_input);

	void ReportLayerFeedSetting(int nStep, int nLayer, int batch_size, double rate);
	void ReportLayerFeedWeight(int nStep, int nLayer, CKDTensor acc_delta, CKDTensor weight);
	void ReportLayerFeedBias(int nStep, int nLayer, CKDTensor acc_delta, CKDTensor bias);
	void ReportLayerFeedBias(int nStep, int nLayer, double acc_delta, CKDTensor bias);

	void ReportFinalResult(int nLayer, CKDTensor weight, CKDTensor bias);
	void ReportFinalResult(int nLayer, CKDTensor weight1, CKDTensor weight2, CKDTensor bias);

	void ReportTestStart(bool test);
	void ReportTestCase(CKDReportInfo &info); // bool test, int nth, CKDTensor input, CKDTensor output, CKDTensor output_est, bool is_correct);
	void ReportTestResult(CKDReportInfo& info); // bool test, int nTestCount, int nCorrectCount);

protected:
	int cost_cnt;
	double cost_sum;
};
