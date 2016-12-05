#include "stdafx.h"

#include "../dataset/dataset.h"
#include "../project/project_manager.h"
#include "../math/tensor.h"
#include "../util/util.h"
#include "../util/xml_util.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

CKDDataSet* CKDDataSet::Create(CKDProjectManager* pConfig, XMLNode* node) {
	XMLNode* dataset_node = CKDXmlUtil::SeekDefClause(node, "dataset");

	if (!dataset_node) return NULL;

	const char* source = CKDXmlUtil::GetAttribute(dataset_node, "source");
	const char* mode = CKDXmlUtil::GetAttribute(dataset_node, "mode");

	CKDDataSet* pDataSet = NULL;

	if (mode == NULL) {
		if (source == NULL) {
			pDataSet = new CKDClosedDataSet(dataset_node);
		}
		else {
			pDataSet = new CKDLoadedDataSet(pConfig, dataset_node, source);
		}
	}
	else if (strcmp(mode, "mnist") == 0) pDataSet = new CKDMnistDataSet(node, source);
	else if (strcmp(mode, "cifar10") == 0) pDataSet = new CKDCifar10DataSet(node, source);
	else if (strcmp(mode, "embedded-reber") == 0) pDataSet = new CKDEmbeddedReberDataSet(node);
	else if (strcmp(mode, "acc_plus_1") == 0) pDataSet = new CKDAccPlusOneDataSet(node);
	else THROW_TEMP;

	pDataSet->SplitTestSet(CKDXmlUtil::SeekTagChild(dataset_node, "split"));
	//const char* style = CKDXmlUtil::SeekTagChild(dataset_node, "style")->ToElement()->GetText();

	//if (strcmp(style, "closed_set") == 0) return new CKDClosedDataSet(dataset_node);

	return pDataSet;
}

CKDDataSet::CKDDataSet(XMLNode* node)
{
	m_nExampleCount = 0;

	m_nTrainExampleCount = 0;
	m_nTestExampleCount = 0;

	m_pTrainIndexes = NULL;
	m_pTestIndexes = NULL;

	m_pExamples = NULL;

	XMLNode* input = CKDXmlUtil::SeekTagChild(node, "input");
	XMLNode* output = CKDXmlUtil::SeekTagChild(node, "output");

	XMLNode* vector = CKDXmlUtil::SeekTagChild(input, "vector");
	XMLNode* grid = CKDXmlUtil::SeekTagChild(input, "grid");

	if (vector) {
		m_inputDim = CKDDimension(1, CKDXmlUtil::GetIntAttr(vector, "size"));
	}
	else if (grid) {
		m_inputDim = CKDDimension(2, CKDXmlUtil::GetIntAttr(grid, "width"), CKDXmlUtil::GetIntAttr(grid, "height"));
	}

	m_outputDim = CKDDimension(1, CKDXmlUtil::GetIntAttr(output, "size"));
}

CKDDataSet::~CKDDataSet()
{
	delete[] m_pTrainIndexes;
	delete[] m_pExamples;
}

void CKDDataSet::SplitTestSet(XMLNode* split) {
	if (split == NULL) return;

	const char* method = CKDXmlUtil::GetAttribute(split, "method");

	if (strcmp(method, "random_exclusive") == 0) {
		double test_ratio = CKDXmlUtil::GetDoubleAttr(split, "test_ratio");

		m_pTrainIndexes = new int[m_nExampleCount];

		for (int i = 0; i < m_nExampleCount; i++) {
			m_pTrainIndexes[i] = (rand() % 100) * m_nExampleCount + i;
		}

		qsort(m_pTrainIndexes, m_nExampleCount, sizeof(int), ms_compare);

		for (int i = 0; i < m_nExampleCount; i++) {
			m_pTrainIndexes[i] = m_pTrainIndexes[i] % m_nExampleCount;
		}

		m_nTestExampleCount = (int)(m_nExampleCount * test_ratio);
		m_nTrainExampleCount = m_nExampleCount - m_nTestExampleCount;

		m_pTestIndexes = m_pTrainIndexes + m_nTrainExampleCount;
	}
	else {
		THROW_TEMP;
	}
}

int CKDDataSet::ms_compare(const void* p1, const void* p2) {
	return *(int*)p1 - *(int*)p2;
}

bool CKDDataSet::GetNthTrainExample(int nth, CKDTensor& input, CKDTensor& output) const {
	if (nth < 0 || nth >= m_nTrainExampleCount) return false;

	input.SetDimension(m_inputDim);
	output.SetDimension(m_outputDim);

	int nDataIndex = m_pTrainIndexes ? m_pTrainIndexes[nth] : nth;

	double* data = m_pExamples + nDataIndex * (m_inputDim.getSize() + m_outputDim.getSize());

	input.SetData(data);
	output.SetData(data + m_inputDim.getSize());

	//pInput = input;
	//pOutput = output;

	return true;
}

bool CKDDataSet::GetNthTestExample(int nth, CKDTensor& input, CKDTensor& output) const {
	if (nth < 0 || nth >= m_nTestExampleCount) return false;

	input.SetDimension(m_inputDim);
	output.SetDimension(m_outputDim);

	int nDataIndex = m_pTestIndexes ? m_pTestIndexes[nth] : nth;

	double* data = m_pExamples + nDataIndex * (m_inputDim.getSize() + m_outputDim.getSize());

	input.SetData(data);
	output.SetData(data + m_inputDim.getSize());

	return true;
}

CKDClosedDataSet::CKDClosedDataSet(XMLNode* node)
: CKDDataSet(node)
{
	int termSize = m_inputDim.getSize() + m_outputDim.getSize();

	XMLNode* data_node = CKDXmlUtil::SeekTagChild(node, "data");

	m_nExampleCount = CKDXmlUtil::GetChildCount(data_node);

	int size = m_nExampleCount * termSize;

	m_pExamples = new double[size];

	memset(m_pExamples, 0, sizeof(double)*size);

	int n = 0;

	for (XMLNode* child = data_node->FirstChild(); child; child = child->NextSibling()) {
		double* pData = m_pExamples + termSize * n++;

		CKDNumbers numbers(child->ToElement()->GetText());
		numbers.GetData(pData, m_inputDim.getSize() + m_outputDim.getSize());
	}

	m_nTrainExampleCount = m_nExampleCount;
	m_nTestExampleCount = m_nExampleCount;
}

CKDClosedDataSet::~CKDClosedDataSet() {
}

CKDLoadedDataSet::CKDLoadedDataSet(CKDProjectManager* pConfig, XMLNode* node, const char* source)
: CKDDataSet(node) {
	XMLNode* output = CKDXmlUtil::SeekTagChild(node, "output");

	CKDBinaryFilters filters(output);
	CKDClassifyClasses classes(output);

	FILE* fid = NULL;

	int lines = 0;
	char buf[1024];

	_getcwd(buf, 1024);

	if (fopen_s(&fid, source, "rt") != 0) THROW_TEMP;

	while (fgets(buf, 1024, fid)) lines++;
	fseek(fid, 0, SEEK_SET);

	int termSize = m_inputDim.getSize() + m_outputDim.getSize();

	//m_dataCount = CKDXmlUtil::GetChildCount(data_node);
	//int size = m_dataCount *termSize;

	m_pExamples = new double[lines*termSize];

	memset(m_pExamples, 0, sizeof(double)*lines*termSize);

	int nLostCount = 0;

	while (fgets(buf, 1024, fid)) {
		int base = termSize * m_nExampleCount;
		CKDUtilTokens tokens(buf);
		for (int i = 0; i < m_inputDim.getSize(); i++) {
			m_pExamples[base + i] = atof(tokens[i]);
		}
		
		enum enumProjectMode mode = pConfig->GetProjectMode();

		if (mode == proj_mode_binary) {
			int output;
			bool matched = filters.Match(tokens[m_inputDim.getSize()], &output);
			if (matched) {
				m_pExamples[base + m_inputDim.getSize()] = output;
				m_nExampleCount++;
			}
			else {
				nLostCount++;
			}
		}
		else if (mode == proj_mode_classify) {
			int index;
			bool matched = classes.Match(tokens[m_inputDim.getSize()], &index);
			if (matched) {
				for (int i = 0; i < m_outputDim.getSize(); i++) {
					m_pExamples[base + m_inputDim.getSize() + i] = (i == index) ? 1 : 0;
				}
				m_nExampleCount++;
			}
			else {
				nLostCount++;
			}
		}
	}

	fclose(fid);

	m_nTrainExampleCount = m_nExampleCount;
	m_nTestExampleCount = m_nExampleCount;
}

CKDLoadedDataSet::~CKDLoadedDataSet() {
}

int CKDMnistDataSet::m_readIntMSB(FILE* fid) {
	int nRead = 0;
	int ndata = 0;
	char* p = (char*)&nRead;
	char* q = (char*)&ndata + sizeof(int)-1;
	if (fread(&nRead, sizeof(int), 1, fid) != 1) THROW_TEMP;

#ifdef _LINUX_
	for (int i = 0; i < sizeof(int); i++) {
		*(q--) = *(p++);
	}

	return ndata;
	//return nRead;
#else
	for (int i = 0; i < sizeof(int); i++) {
		*(q--) = *(p++);
	}

	return ndata;
#endif
}

CKDMnistDataSet::CKDMnistDataSet(XMLNode* node, const char* source)
: CKDDataSet(node) {
	m_pTrainInputPool = NULL;
	m_pTrainOutputPool = NULL;
	m_pTestInputPool = NULL;
	m_pTestOutputPool = NULL;

	FILE* fid = NULL;
	char path[1024];

	sprintf_s(path, 1024, "%s%ctrain-images.idx3-ubyte", source, PATH_DELIMETER);
	if (fopen_s(&fid, path, "rb") != 0) THROW_TEMP;
	if (m_readIntMSB(fid) != 0x803) THROW_TEMP;
	m_nTrainExampleCount = m_readIntMSB(fid);
	
	int nRows = m_readIntMSB(fid);
	int nCols = m_readIntMSB(fid);
		
	m_pTrainInputPool = new unsigned char[m_nTrainExampleCount*nRows*nCols];

	if (fread(m_pTrainInputPool, nRows*nCols, m_nTrainExampleCount, fid) != m_nTrainExampleCount) THROW_TEMP;

	fclose(fid);

	sprintf_s(path, 1024, "%s%ctrain-labels.idx1-ubyte", source, PATH_DELIMETER);

	if (fopen_s(&fid, path, "rb") != 0) THROW_TEMP;
	if (m_readIntMSB(fid) != 0x801) THROW_TEMP;
	m_nTrainExampleCount = m_readIntMSB(fid);

	m_pTrainOutputPool = new unsigned char[m_nTrainExampleCount];

	if (fread(m_pTrainOutputPool, 1, m_nTrainExampleCount, fid) != m_nTrainExampleCount) THROW_TEMP;

	fclose(fid);

	sprintf_s(path, 1024, "%s%ct10k-images.idx3-ubyte", source, PATH_DELIMETER);

	if (fopen_s(&fid, path, "rb") != 0) THROW_TEMP;
	if (m_readIntMSB(fid) != 0x803) THROW_TEMP;
	m_nTestExampleCount = m_readIntMSB(fid);
	
	nRows = m_readIntMSB(fid);
	nCols = m_readIntMSB(fid);

	m_pTestInputPool = new unsigned char[m_nTestExampleCount*nRows*nCols];

	if (fread(m_pTestInputPool, nRows*nCols, m_nTestExampleCount, fid) != m_nTestExampleCount) THROW_TEMP;

	fclose(fid);

	sprintf_s(path, 1024, "%s%ct10k-labels.idx1-ubyte", source, PATH_DELIMETER);

	if (fopen_s(&fid, path, "rb") != 0) THROW_TEMP;
	if (m_readIntMSB(fid) != 0x801) THROW_TEMP;
	m_nTestExampleCount = m_readIntMSB(fid);

	m_pTestOutputPool = new unsigned char[m_nTestExampleCount];

	if (fread(m_pTestOutputPool, 1, m_nTestExampleCount, fid) != m_nTestExampleCount) THROW_TEMP;

	fclose(fid);

	m_inputDim.setSize(nRows, nCols);
	m_outputDim.setSize(10);
}

CKDMnistDataSet::~CKDMnistDataSet() {
	delete [] m_pTrainInputPool;
	delete [] m_pTrainOutputPool;
	delete [] m_pTestInputPool;
	delete [] m_pTestOutputPool;
}

bool CKDMnistDataSet::GetNthTrainExample(int nth, CKDTensor& input, CKDTensor& output) const {
	if (nth < 0 || nth >= m_nTrainExampleCount) return false;

	input.SetDimension(m_inputDim);
	output.SetDimension(m_outputDim);

	unsigned char* p = m_pTrainInputPool + nth * (m_inputDim.getSize());

	input.SetDataToDouble(p);
	output[m_pTrainOutputPool[nth]] = 1;

	return true;
}

bool CKDMnistDataSet::GetNthTestExample(int nth, CKDTensor& input, CKDTensor& output) const {
	if (nth < 0 || nth >= m_nTestExampleCount) return false;

	input.SetDimension(m_inputDim);
	output.SetDimension(m_outputDim);

	unsigned char* p = m_pTestInputPool + nth * (m_inputDim.getSize());

	input.SetDataToDouble(p);
	output[m_pTestOutputPool[nth]] = 1;

	return true;
}

CKDCifar10DataSet::CKDCifar10DataSet(XMLNode* node, const char* source)
: CKDDataSet(node) {
	m_pTrainPool = NULL;
	m_pTestPool = NULL;

	FILE* fid = NULL;
	char path[1024];

	int nRows = 32;
	int nCols = 32;
	int nPlanes = 3;

	int data_size = nRows*nCols*nPlanes + 1;

	m_nTrainExampleCount = 50000;
	m_nTestExampleCount = 10000;

	m_pTrainPool = new unsigned char[m_nTrainExampleCount*data_size];
	m_pTestPool = new unsigned char[m_nTestExampleCount*data_size];

	for (int i = 0; i < 5; i++) {
		sprintf_s(path, 1024, "%s%cdata_batch_%d.bin", source, PATH_DELIMETER, i+1);

		if (fopen_s(&fid, path, "rb") != 0) THROW_TEMP;

		unsigned char* pBuf = m_pTrainPool + i * data_size * 10000;

		if (fread(pBuf, data_size, 10000, fid) != 10000) THROW_TEMP;

		fclose(fid);
	}

	sprintf_s(path, 1024, "%s%ctest_batch.bin", source, PATH_DELIMETER);

	if (fopen_s(&fid, path, "rb") != 0) THROW_TEMP;
	
	if (fread(m_pTestPool, data_size, m_nTestExampleCount, fid) != m_nTestExampleCount) THROW_TEMP;

	fclose(fid);

	m_inputDim.setSize(nRows, nCols, 3);
	m_outputDim.setSize(10);
}

CKDCifar10DataSet::~CKDCifar10DataSet() {
	delete [] m_pTrainPool;
	delete [] m_pTestPool;
}

bool CKDCifar10DataSet::GetNthTrainExample(int nth, CKDTensor& input, CKDTensor& output) const {
	if (nth < 0 || nth >= m_nTrainExampleCount) return false;

	input.SetDimension(m_inputDim);
	output.SetDimension(m_outputDim);

	unsigned char* p = m_pTrainPool + nth * (m_inputDim.getSize()+1);

	output[p[0]] = 1;
	m_rearrange_data_buffer(input, p+1);

	return true;
}

bool CKDCifar10DataSet::GetNthTestExample(int nth, CKDTensor& input, CKDTensor& output) const {
	if (nth < 0 || nth >= m_nTestExampleCount) return false;

	input.SetDimension(m_inputDim);
	output.SetDimension(m_outputDim);

	const unsigned char* p = m_pTestPool + nth * (m_inputDim.getSize()+1);

	output[p[0]] = 1;
	m_rearrange_data_buffer(input, p + 1);

	return true;
}

void CKDCifar10DataSet::m_rearrange_data_buffer(CKDTensor& input, const unsigned char* pBuf) const {
	int rows = m_inputDim.getAxisSize(0);
	int cols = m_inputDim.getAxisSize(1);
	int planes = m_inputDim.getAxisSize(2);

	for (int np = 0; np < planes; np++) {
		for (int nr = 0; nr < rows; nr++) {
			for (int nc = 0; nc < cols; nc++) {
				input.set(nr, nc, np, *pBuf++);
			}
		}
	}
}

CKDEmbeddedReberDataSet::CKDEmbeddedReberDataSet(XMLNode* node)
: CKDDataSet(node) {
	m_nTrainExampleCount = 2;	// one random correct, one random incorrect
	m_nTestExampleCount = 2;	// one random correct, one random incorrect

	m_inputDim.setSize(7);	// B-PSTVX-E
	m_outputDim.setSize(7);	// PSTVX-E-0(Err)
}

CKDEmbeddedReberDataSet::~CKDEmbeddedReberDataSet() {
}

bool CKDEmbeddedReberDataSet::GetNthTrainExample(int nth, CKDTensor& input, CKDTensor& output) const {
	m_generateSentence(nth, input, output);
	return true;
}

bool CKDEmbeddedReberDataSet::GetNthTestExample(int nth, CKDTensor& input, CKDTensor& output) const {
	m_generateSentence(nth, input, output);
	return true;
}

void CKDEmbeddedReberDataSet::m_generateSentence(int nth, CKDTensor& input, CKDTensor& output) const {
	int state = 0;
	int choice;
	int alphabets[100];
	int idx = 0;

	while (state >= 0) {
		//choice = 0; // rand() % 2;
		choice = rand() % 2;

		switch (state) {
		case 0:
			state = m_move_state(reber_B, 1, alphabets, idx);
			break;
		case 1:
			if (choice == 0) state = m_move_state(reber_T, 2, alphabets, idx);
			else state = m_move_state(reber_P, 3, alphabets, idx);
			break;
		case 2:
			if (choice == 0) state = m_move_state(reber_X, 3, alphabets, idx);
			else state = m_move_state(reber_S, 4, alphabets, idx);
			break;
		case 3:
			if (choice == 0) state = m_move_state(reber_V, 5, alphabets, idx); 
			else state = m_move_state(reber_T, 3, alphabets, idx);
			break;
		case 4:
			if (choice == 0) state = m_move_state(reber_X, 3, alphabets, idx);
			else state = m_move_state(reber_S, 6, alphabets, idx);
			break;
		case 5:
			if (choice == 0) state = m_move_state(reber_V, 6, alphabets, idx);
			else state = m_move_state(reber_P, 4, alphabets, idx); 
			break;
		case 6:
			state = m_move_state(reber_E, -1, alphabets, idx);
			break;
		}
	}

	input = CKDTensor(CKDDimension(2, idx, 7));
	output = CKDTensor(CKDDimension(2, idx, 7));

	for (int i = 0; i < idx; i++) {
		input.set(i, alphabets[i], 1);
	}

	for (int i = 0; i < idx-1; i++) {
		output.set(i, alphabets[i+1], 1);
	}

	output.set(idx - 1, reber_0, 1);
}

int CKDEmbeddedReberDataSet::m_move_state(enum reberAlphabet alphabet, int next_state, int *alphabets, int& idx) const {
	if (idx >= 100) {	// too long sentence, do it again
		idx = 0;
		return 0;
	}
	alphabets[idx++] = alphabet;
	return next_state;
}

CKDAccPlusOneDataSet::CKDAccPlusOneDataSet(XMLNode* node)
: CKDDataSet(node) {
	m_nTrainExampleCount = 1;	// one random correct, one random incorrect
	m_nTestExampleCount = 1;	// one random correct, one random incorrect

	m_inputDim.setSize(1);
	m_outputDim.setSize(1);
}

CKDAccPlusOneDataSet::~CKDAccPlusOneDataSet() {
}

bool CKDAccPlusOneDataSet::GetNthTrainExample(int nth, CKDTensor& input, CKDTensor& output) const {
	m_generateSentence(nth, input, output);
	return true;
}

bool CKDAccPlusOneDataSet::GetNthTestExample(int nth, CKDTensor& input, CKDTensor& output) const {
	m_generateSentence(nth, input, output);
	return true;
}

void CKDAccPlusOneDataSet::m_generateSentence(int nth, CKDTensor& input, CKDTensor& output) const {
	input = CKDTensor(CKDDimension(2, 3, 1));
	output = CKDTensor(CKDDimension(2, 3, 1));

	int sum = 1;

	for (int i = 0; i < 3; i++) {
		int n = rand() % 3;
		sum += n;
		input.set(i, 0, n);
		output.set(i, 0, sum);
	}
}

CKDBinaryFilters::CKDBinaryFilters(XMLNode* node) {
	m_nCount = 0;
	m_pFilters = NULL;

	int count = CKDXmlUtil::GetTagChildCount(node, "filter");
	m_pFilters = new CKDBinaryFilter[count];

	XMLNode* filter = CKDXmlUtil::SeekTagChild(node, "filter");

	while (filter) {
		m_pFilters[m_nCount++].Setup(filter);
		filter = CKDXmlUtil::SeekTagSibling(filter, "filter");
	}
}

CKDBinaryFilters::~CKDBinaryFilters() {
	delete[] m_pFilters;
}

bool CKDBinaryFilters::Match(const char* str, int* pnum) const {
	for (int i = 0; i < m_nCount; i++) {
		if (m_pFilters[i].Match(str, pnum)) return true;
	}

	return false;
}

CKDBinaryFilter::CKDBinaryFilter() {
	m_value = NULL;
	m_result = 0;
}

CKDBinaryFilter::~CKDBinaryFilter() {
}

void CKDBinaryFilter::Setup(XMLNode* filter) {
	m_value = CKDXmlUtil::GetAttribute(filter, "value");
	m_result = atoi(filter->ToElement()->GetText());
}

bool CKDBinaryFilter::Match(const char* value, int* pResult) const {
	if (m_value && strcmp(value, m_value) == 0) {
		*pResult = m_result;
		return true;
	}

	return false;
}

CKDClassifyClasses::CKDClassifyClasses(XMLNode* node) {
	m_nCount = 0;
	m_pClasses = NULL;

	int count = CKDXmlUtil::GetTagChildCount(node, "class");
	m_pClasses = new CKDClassifyClass[count];

	XMLNode* child = CKDXmlUtil::SeekTagChild(node, "class");

	while (child) {
		m_pClasses[m_nCount++].Setup(child);
		child = CKDXmlUtil::SeekTagSibling(child, "class");
	}
}

CKDClassifyClasses::~CKDClassifyClasses() {
	delete[] m_pClasses;
}

bool CKDClassifyClasses::Match(const char* str, int* pnum) const {
	for (int i = 0; i < m_nCount; i++) {
		if (m_pClasses[i].Match(str, pnum)) return true;
	}

	return false;
}

CKDClassifyClass::CKDClassifyClass() {
	m_value = NULL;
	m_index = 0;
}

CKDClassifyClass::~CKDClassifyClass() {
}

void CKDClassifyClass::Setup(XMLNode* filter) {
	m_value = CKDXmlUtil::GetAttribute(filter, "value");
	m_index = CKDXmlUtil::GetIntAttr(filter, "index");
}

bool CKDClassifyClass::Match(const char* value, int* pIndex) const {
	if (m_value && strcmp(value, m_value) == 0) {
		*pIndex = m_index;
		return true;
	}

	return false;
}
