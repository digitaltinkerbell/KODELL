#include "stdafx.h"

#include "time.h"
#include "stdlib.h"
#include "stdio.h"

#include "../project/reporter.h"
#include "../math/tensor.h"
#include "../util/xml_util.h"
#include "../util/util.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

CKDReporter::CKDReporter() {
	m_nReportFlags = 0;
	m_destType = enum_report_dest_none;
	m_fid = NULL;
 	m_nRangeCount = 0;
	m_pRanges = NULL;
}

CKDReporter::~CKDReporter() {
	if (m_fid && m_fid != stdout) fclose(m_fid);
	delete[] m_pRanges;
}

void CKDReporterRange::Setup(XMLNode* node) {
	m_period = CKDXmlUtil::GetIntAttr(node, "period", 1);
	m_offset = CKDXmlUtil::GetIntAttr(node, "offset", 0);
	m_from = CKDXmlUtil::GetIntAttr(node, "from", 0);
	m_to = CKDXmlUtil::GetIntAttr(node, "to", -1);
}

void CKDReporter::Setup(XMLNode* node) {
	/*
	<report type = "full" neglist="step_end" period = "1000" path = "dump\\xor_mono.full.txt">< / report>
	<report type = "brief" period = "100" path = "dump\\xor_mono.brief.txt">< / report>
	<report poslist = "step_start cost" period = "100" dest="stdout">< / report>
	<report poslist = "step_start cost" period = "100" offset="50" dest="callback">< / report>
	*/

	const char* type = CKDXmlUtil::GetAttribute(node, "type");

	if (type) {
		if (strcmp(type, "full") == 0) m_nReportFlags = REPORT_FLAGS_ALL;
		else if (strcmp(type, "brief") == 0) m_nReportFlags = REPORT_FLAGS_BRIEF;
	}

	CKDUtilTokens pos_tokens(CKDXmlUtil::GetAttribute(node, "poslist"));

	for (int i = 0; i < pos_tokens.size(); i++) {
		m_nReportFlags |= m_getTokenMask(pos_tokens[i]);
	}

	CKDUtilTokens neg_tokens(CKDXmlUtil::GetAttribute(node, "neglist"));

	for (int i = 0; i < neg_tokens.size(); i++) {
		m_nReportFlags &= ~m_getTokenMask(neg_tokens[i]);
	}

	m_nRangeCount = CKDXmlUtil::GetTagChildCount(node, "range") + 1;
	m_pRanges = new CKDReporterRange[m_nRangeCount];

	m_pRanges[0].Setup(node);

	XMLNode* child = CKDXmlUtil::SeekTagChild(node, "range");
	for (int i = 1; i < m_nRangeCount; i++) {
		m_pRanges[i].Setup(child);
		child = CKDXmlUtil::SeekTagSibling(child, "range");
	}

	const char* dest = CKDXmlUtil::GetAttribute(node, "dest");
	const char* path = CKDXmlUtil::GetAttribute(node, "path");

	if (!dest) dest = path ? "file" : "stdout";

	if (strcmp(dest, "file") == 0) {
		if (!path) THROW_TEMP;
		m_destType = enum_report_dest_file;
		m_fid = CKDUtil::FileOpen(path, "wt");
	}
	else if (strcmp(dest, "stdout") == 0) {
		m_destType = enum_report_dest_stdout;
		m_fid = stdout;
	}
	else if (strcmp(dest, "callback") == 0) {
		m_destType = enum_report_dest_callback;
	}
	else {
		THROW_TEMP;
	}
}

int CKDReporter::m_getTokenMask(const char* token) {
	if (!token) return 0;

	if (strcmp(token, "debug") == 0) return REPORT_FLAG_DEBUG;

	if (strcmp(token, "train") == 0) return REPORT_FLAG_TRAIN_START | REPORT_FLAG_TRAIN_EXIT | REPORT_FLAG_TRAIN_END;
	if (strcmp(token, "train_start") == 0) return REPORT_FLAG_TRAIN_START;
	if (strcmp(token, "train_exit")==0) return REPORT_FLAG_TRAIN_EXIT;
	if (strcmp(token, "train_end")==0) return REPORT_FLAG_TRAIN_END;

	if (strcmp(token, "step")==0) return REPORT_FLAG_STEP_START | REPORT_FLAG_STEP_END;
	if (strcmp(token, "step_start")==0) return REPORT_FLAG_STEP_START;
	if (strcmp(token, "step_end")==0) return REPORT_FLAG_STEP_END;

	if (strcmp(token, "minibatch")==0) return REPORT_FLAG_MINIBATCH_SELECT | REPORT_FLAG_MINIBATCH_VOMIT;
	if (strcmp(token, "minibatch_select")==0) return REPORT_FLAG_MINIBATCH_SELECT;
	if (strcmp(token, "minibatch_vomit")==0) return REPORT_FLAG_MINIBATCH_VOMIT;

	if (strcmp(token, "example")==0) return REPORT_FLAG_EXAMPLE;

	if (strcmp(token, "evaluate")==0) return REPORT_FLAG_LAYER_LINEAR_EVAL | REPORT_FLAG_LAYER_ACTIVATE_EVAL;
	if (strcmp(token, "linear")==0) return REPORT_FLAG_LAYER_LINEAR_EVAL;
	if (strcmp(token, "activate")==0) return REPORT_FLAG_LAYER_ACTIVATE_EVAL;

	if (strcmp(token, "cost") == 0) return REPORT_FLAG_COST;
	if (strcmp(token, "cost_avg") == 0) return REPORT_FLAG_COST_AVG;
	if (strcmp(token, "grad") == 0) return REPORT_FLAG_COST_GRAD;

	if (strcmp(token, "delta")==0) return REPORT_FLAG_LAYER_GRAD_DELTA | REPORT_FLAG_LAYER_BIAS_DELTA | REPORT_FLAG_LAYER_INPUT_GRAD;
	if (strcmp(token, "grad_delta")==0) return REPORT_FLAG_LAYER_GRAD_DELTA;
	if (strcmp(token, "bias_delta")==0) return REPORT_FLAG_LAYER_BIAS_DELTA;
	if (strcmp(token, "input_delta")==0) return REPORT_FLAG_LAYER_INPUT_GRAD;

	if (strcmp(token, "feed")==0) return REPORT_FLAG_LAYER_FEED_SETTING | REPORT_FLAG_LAYER_FEED_WEIGHT | REPORT_FLAG_LAYER_FEED_BIAS;
	if (strcmp(token, "feed_setting")==0) return REPORT_FLAG_LAYER_FEED_SETTING;
	if (strcmp(token, "feed_weight")==0) return REPORT_FLAG_LAYER_FEED_WEIGHT;
	if (strcmp(token, "feed_bias")==0) return REPORT_FLAG_LAYER_FEED_BIAS;

	if (strcmp(token, "final") == 0) return REPORT_FLAG_FINAL_RESULT;
	if (strcmp(token, "result") == 0) return REPORT_FLAG_FINAL_RESULT;

	if (strcmp(token, "test") == 0) return REPORT_FLAG_TEST_START | REPORT_FLAG_TEST_CASE | REPORT_FLAG_TEST_RESULT;
	if (strcmp(token, "test_start") == 0) return REPORT_FLAG_TEST_START;
	if (strcmp(token, "test_case") == 0) return REPORT_FLAG_TEST_CASE;
	if (strcmp(token, "test_result") == 0) return REPORT_FLAG_TEST_RESULT;

	return 0;
}

bool CKDReporter::m_stepCheck(int step) const {
	for (int i = 0; i < m_nRangeCount; i++) {
		if ( m_pRanges[i].m_stepCheck(step)) return true;
	}

	return false;
}
 
void CKDReporter::ReportDebug(const char* caption, const CKDTensor& tensor) {
	if (!m_flagCheck(REPORT_FLAG_DEBUG)) return;

	if (m_fid) {
		tensor.dump(m_fid, caption);
		fflush(m_fid);
	}
}

void CKDReporter::ReportTrainStart(const char* pProjectName) {
	if (!m_flagCheck(REPORT_FLAG_TRAIN_START)) return;

	if (m_fid) {
		fprintf(m_fid, "[TRAIN %s START] %d\n\n", pProjectName, (int) time(NULL));
		fflush(m_fid);
	}
}

void CKDReporter::ReportTrainExit(const char* pReason) {
	if (!m_flagCheck(REPORT_FLAG_TRAIN_EXIT)) return;

	if (m_fid) {
		fprintf(m_fid, "[TRAIN exit: %s] %d\n\n", pReason, (int) time(NULL));
		fflush(m_fid);
	}
}

void CKDReporter::ReportTrainEnd(int nStep) {
	if (!m_flagCheck(REPORT_FLAG_TRAIN_END)) return;

	if (m_fid) {
		fprintf(m_fid, "[TRAIN END] in %d steps, time=%d\n\n", nStep, (int) time(NULL));
		fflush(m_fid);
	}
}

void CKDReporter::ReportIterateStepStart(int nStep) {
	if (!m_flagCheck(REPORT_FLAG_STEP_START)) return;
	if (!m_stepCheck(nStep)) return;

	if (m_fid) {
		fprintf(m_fid, "[STEP %d START] %d\n", nStep, (int) time(NULL));
		fflush(m_fid);
	}
}

void CKDReporter::ReportIterateStepEnd(int nStep) {
	if (!m_flagCheck(REPORT_FLAG_STEP_END)) return;
	if (!m_stepCheck(nStep)) return;

	if (m_fid) {
		fprintf(m_fid, "[STEP %d END] %d\n", nStep, (int) time(NULL));
		fflush(m_fid);
	}
}

void CKDReporter::ReportMinibatchSelect(int nStep, int nth, int src_idx) {
	if (!m_flagCheck(REPORT_FLAG_MINIBATCH_SELECT)) return;
	if (!m_stepCheck(nStep)) return;

	if (m_fid) {
		fprintf(m_fid, "\tMinibatch[%d] was assigned as Dataset[%d]\n", nth, src_idx);
		fflush(m_fid);
	}
}

void CKDReporter::ReportMinibatchVomit(int nStep, int nth) {
	if (!m_flagCheck(REPORT_FLAG_MINIBATCH_VOMIT)) return;
	if (!m_stepCheck(nStep)) return;

	if (m_fid) {
		fprintf(m_fid, "\tMinibatch[%d] assign was canceled\n", nth);
		fflush(m_fid);
	}
}

void CKDReporter::ReportExample(int nStep, CKDTensor input, CKDTensor output) {
	if (!m_flagCheck(REPORT_FLAG_EXAMPLE)) return;
	if (!m_stepCheck(nStep)) return;

	if (m_fid) {
		input.dump(m_fid, "*** example input");
		output.dump(m_fid, "*** example output");
		fflush(m_fid);
	}
}

void CKDReporter::ReportSetLayer(int nStep, int nLayer, int nLayerCount, enum enumTrainPhase trainPhase) {
	if (!m_flagCheck(REPORT_FLAG_SET_LAYER)) return;
	if (!m_stepCheck(nStep)) return;
}

void CKDReporter::ReportLayerLinearEval(int nStep, int nLayer, CKDTensor weight, CKDTensor bias, CKDTensor net) {
	if (!m_flagCheck(REPORT_FLAG_LAYER_LINEAR_EVAL)) return;
	if (!m_stepCheck(nStep)) return;

	if (m_fid) {
		char buf[256];
		sprintf_s(buf, 256, "weight-L%d", nLayer);
		weight.dump(m_fid, buf);
		sprintf_s(buf, 256, "bias-L%d", nLayer);
		bias.dump(m_fid, buf);
		sprintf_s(buf, 256, "net-L%d", nLayer);
		net.dump(m_fid, buf);
		fflush(m_fid);
	}
}

void CKDReporter::ReportLayerActivateEval(int nStep, int nLayer, enum actFuncType func, CKDTensor out) {
	if (!m_flagCheck(REPORT_FLAG_LAYER_ACTIVATE_EVAL)) return;
	if (!m_stepCheck(nStep)) return;

	if (m_fid) {
		fprintf(m_fid, "\tactivate_function-L%d: %s\n", nLayer, CKDUtil::GetActFuncName(func));
		char buf[256];
		sprintf_s(buf, 256, "out-L%d", nLayer);
		out.dump(m_fid, buf);
		fflush(m_fid);
	}
}

void CKDReporter::ReportCost(int nStep, double cost, bool reset, bool close) {
	if (!m_stepCheck(nStep)) return;
	if (reset) {
		cost_cnt = 0;
		cost_sum = 0;

	}

	cost_cnt++;
	cost_sum += cost;

	if (m_flagCheck(REPORT_FLAG_COST)) {
		if (m_fid) {
			fprintf(m_fid, "\tcost = %11.9lf\n", cost);
			fflush(m_fid);
		}
	}
	else if (close && m_flagCheck(REPORT_FLAG_COST_AVG)) {
		if (m_fid) {
			fprintf(m_fid, "\tcost_avg = %11.9lf\n", cost_sum / cost_cnt);
			fflush(m_fid);
		}
	}
}

void CKDReporter::ReportCostGradient(int nStep, CKDTensor grad) {
	if (!m_flagCheck(REPORT_FLAG_COST_GRAD)) return;
	if (!m_stepCheck(nStep)) return;

	if (m_fid) {
		grad.dump(m_fid, "cost_gradient");
		fflush(m_fid);
	}
}

void CKDReporter::ReportLayerGradient(int nStep, int nLayer, CKDTensor grad) {
	if (!m_flagCheck(REPORT_FLAG_LAYER_GRAD)) return;
	if (!m_stepCheck(nStep)) return;

	if (m_fid) {
		char buf[256];
		sprintf_s(buf, 256, "layer_gradient-L%d", nLayer);
		grad.dump(m_fid, buf);
		fflush(m_fid);
	}
}

void CKDReporter::ReportLayerWeightGradDelta(int nStep, int nLayer, CKDTensor delta, CKDTensor acc) {
	if (!m_flagCheck(REPORT_FLAG_LAYER_GRAD_DELTA)) return;
	if (!m_stepCheck(nStep)) return;

	if (m_fid) {
		char buf[256];
		sprintf_s(buf, 256, "weight_delta-L%d", nLayer);
		delta.dump(m_fid, buf);
		sprintf_s(buf, 256, "weight_delta_acc-L%d", nLayer);
		acc.dump(m_fid, buf);
		fflush(m_fid);
	}
}

void CKDReporter::ReportLayerBiasGradDelta(int nStep, int nLayer, CKDTensor delta, CKDTensor acc) {
	if (!m_flagCheck(REPORT_FLAG_LAYER_BIAS_DELTA)) return;
	if (!m_stepCheck(nStep)) return;

	if (m_fid) {
		char buf[256];
		sprintf_s(buf, 256, "bias_delta-L%d", nLayer);
		delta.dump(m_fid, buf);
		sprintf_s(buf, 256, "bias_delta_acc-L%d", nLayer);
		acc.dump(m_fid, buf);
		fflush(m_fid);
	}
}

void CKDReporter::ReportLayerBiasGradDelta(int nStep, int nLayer, double delta, double acc) {
	if (!m_flagCheck(REPORT_FLAG_LAYER_BIAS_DELTA)) return;
	if (!m_stepCheck(nStep)) return;

	if (m_fid) {
		fprintf(m_fid, "\tbias_delta-L%d: %9.6f\n", nLayer, delta);
		fprintf(m_fid, "\tbias_delta_acc-L%d: %9.6f\n", nLayer, acc);
		fflush(m_fid);
	}
}

void CKDReporter::ReportLayerInputGrad(int nStep, int nLayer, CKDTensor delta_input) {
	if (!m_flagCheck(REPORT_FLAG_LAYER_INPUT_GRAD)) return;
	if (!m_stepCheck(nStep)) return;

	if (m_fid) {
		char buf[256];
		sprintf_s(buf, 256, "input_grad-L%d", nLayer);
		delta_input.dump(m_fid, buf);
		fflush(m_fid);
	}
}

void CKDReporter::ReportLayerFeedSetting(int nStep, int nLayer, int batch_size, double rate) {
	if (!m_flagCheck(REPORT_FLAG_LAYER_FEED_SETTING)) return;
	if (!m_stepCheck(nStep)) return;

	if (m_fid) {
		fprintf(m_fid, "\tbatch_size-L%d: %d\n", nLayer, batch_size);
		fprintf(m_fid, "\tlearning_rate-L%d: %9.6f\n", nLayer, rate);
		fflush(m_fid);
	}
}

void CKDReporter::ReportLayerFeedWeight(int nStep, int nLayer, CKDTensor acc_delta, CKDTensor weight) {
	if (!m_flagCheck(REPORT_FLAG_LAYER_FEED_WEIGHT)) return;
	if (!m_stepCheck(nStep)) return;

	if (m_fid) {
		char buf[256];
		sprintf_s(buf, 256, "feed_weight_delta-L%d", nLayer);
		acc_delta.dump(m_fid, buf);
		sprintf_s(buf, 256, "feed_weight_result-L%d", nLayer);
		weight.dump(m_fid, buf);
		fflush(m_fid);
	}
}

void CKDReporter::ReportLayerFeedBias(int nStep, int nLayer, CKDTensor acc_delta, CKDTensor bias) {
	if (!m_flagCheck(REPORT_FLAG_LAYER_FEED_BIAS)) return;
	if (!m_stepCheck(nStep)) return;

	if (m_fid) {
		char buf[256];
		sprintf_s(buf, 256, "feed_bias_delta-L%d", nLayer);
		acc_delta.dump(m_fid, buf);
		sprintf_s(buf, 256, "feed_bias_result-L%d", nLayer);
		bias.dump(m_fid, buf);
		fflush(m_fid);
	}
}

void CKDReporter::ReportLayerFeedBias(int nStep, int nLayer, double acc_delta, CKDTensor bias) {
	if (!m_flagCheck(REPORT_FLAG_LAYER_FEED_BIAS)) return;
	if (!m_stepCheck(nStep)) return;

	if (m_fid) {
		fprintf(m_fid, "\tfeed_bias_delta-L%d: %9.6f\n", nLayer, acc_delta);
		char buf[256];
		sprintf_s(buf, 256, "feed_bias_result-L%d", nLayer);
		bias.dump(m_fid, buf);
		fflush(m_fid);
	}
}

void CKDReporter::ReportFinalResult(int nLayer, CKDTensor weight, CKDTensor bias) {
	if (!m_flagCheck(REPORT_FLAG_FINAL_RESULT)) return;

	if (m_fid) {
		char buf[256];
		sprintf_s(buf, 256, "FINAL-WEIGHT-L%d", nLayer);
		weight.dump(m_fid, buf);
		sprintf_s(buf, 256, "FINAL-BIAS-L%d", nLayer);
		bias.dump(m_fid, buf);
		fflush(m_fid);
	}
}

void CKDReporter::ReportFinalResult(int nLayer, CKDTensor weight1, CKDTensor weight2, CKDTensor bias) {
	if (!m_flagCheck(REPORT_FLAG_FINAL_RESULT)) return;

	if (m_fid) {
		char buf[256];
		sprintf_s(buf, 256, "FINAL-WEIGHT1-L%d", nLayer);
		weight1.dump(m_fid, buf);
		sprintf_s(buf, 256, "FINAL-WEIGHT2-L%d", nLayer);
		weight2.dump(m_fid, buf);
		sprintf_s(buf, 256, "FINAL-BIAS-L%d", nLayer);
		bias.dump(m_fid, buf);
		fflush(m_fid);
	}
}

void CKDReporter::ReportTestStart(bool test) {
	if (!m_flagCheck(REPORT_FLAG_TEST_START)) return;

	if (m_fid) {
		fprintf(m_fid, "\t*** TEST PHASE STARTED FOR %s SET ***\n", test ? "TEST" : "TRAIN");
		fflush(m_fid);
	}
}

// void CKDReporter::ReportTestCase(bool test, int nth, CKDTensor input, CKDTensor output, CKDTensor output_est, bool is_correct) {
void CKDReporter::ReportTestCase(CKDReportInfo &info) {
	if (!m_flagCheck(REPORT_FLAG_TEST_CASE)) return;

	int nth = info.getValue("nth");
	bool test = info.getValue("is_test") != 0;
	bool is_correct = info.getValue("is_correct") != 0;

	if (m_fid) {
		fprintf(m_fid, "\t%s Example %d: %s\n", test ? "TEST" : "TRAIN", nth, is_correct ? "correct" : "incorrect");
		//input.dump(m_fid, "input");
		//output.dump(m_fid, "output");
		//output_est.dump(m_fid, "output_est");
		fflush(m_fid);
	}
}

//void CKDReporter::ReportTestResult(bool test, int nTestCount, int nCorrectCount) {
void CKDReporter::ReportTestResult(CKDReportInfo &info) {
	if (!m_flagCheck(REPORT_FLAG_TEST_RESULT)) return;

	bool test = info.getValue("is_test") != 0;
	int nCorrectCount = info.getValue("correct_count");
	int nTestCount = info.getValue("test_count");

	if (m_fid) {
		fprintf(m_fid, "\tCorrectness Ratio: %d / %d = %5.3lf\n", nCorrectCount, nTestCount, (double)nCorrectCount/nTestCount);
		fprintf(m_fid, "\t*** %s PHASE STARTED ***\n", test ? "TEST" : "TRAIN");
		fflush(m_fid);
	}
}

CKDReportInfo::CKDReportInfo() {
	m_pMapHead = NULL;
	m_pMapTail = NULL;
}

CKDReportInfo::~CKDReportInfo() {
	delete m_pMapHead;
}

void CKDReportInfo::setValue(const char* pName, int value) {
	CKDReportNode* pNode = new CKDReportNode(pName, value);
	if (m_pMapHead) m_pMapTail->m_pNextNode = pNode;
	else m_pMapHead = pNode;
	m_pMapTail = pNode;
}

int CKDReportInfo::getValue(const char* pName) {
	CKDReportNode* pNode = m_pMapHead->seekNode(pName);
	if (pNode == NULL) THROW_TEMP;
	return pNode->m_value;
}

CKDReportNode::CKDReportNode(const char* pName, int value) {
	m_pName = pName;
	m_value = value;
	m_pNextNode = NULL;
}

CKDReportNode::~CKDReportNode() {
	delete m_pNextNode;
}

CKDReportNode* CKDReportNode::seekNode(const char* pName) {
	if (this == NULL) return NULL;
	if (strcmp(m_pName, pName) == 0) return this;
	return m_pNextNode->seekNode(pName);
}
