#include "stdafx.h"

#include "string.h"

#include "../config/types.h"
#include "../util/util.h"
#include "../math/tensor.h"
#include "../engine/layer.h"
#include "../project/network_model.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

CKDNumbers::CKDNumbers() {
	m_nCount = 0;
	m_pValues = NULL;
}

CKDNumbers::CKDNumbers(const char* pText) {
	m_nCount = 0;
	m_pValues = NULL;

	SetData(pText);
}

CKDNumbers::~CKDNumbers() {
	delete[] m_pValues;
}

void CKDNumbers::SetData(const char* pText) {
	if (m_pValues) {
		delete[] m_pValues;

		m_nCount = 0;
		m_pValues = NULL;
	}

	int nLength = strlen(pText);
	char* pBuffer = new char[nLength + 1];
	double* pValues = new double[nLength];
	int nCount = 0;

	strcpy_s(pBuffer, nLength+1, pText);

	char* context;
	char* data = strtok_s(pBuffer, ", ", &context);
	while (data) {
		pValues[nCount++] = atof(data);
		data = strtok_s(NULL, ", ", &context);
	}

	m_nCount = nCount;
	m_pValues = new double[m_nCount];

	memcpy(m_pValues, pValues, sizeof(double)*m_nCount);

	delete[] pBuffer;
	delete[] pValues;
}

void CKDNumbers::SetData(XMLNode* pNode) {
	if (pNode == NULL) return;
	SetData(pNode->ToElement()->GetText());
}

void CKDNumbers::GetData(double* pDest, int nMaxSize) const {
	if (nMaxSize <= m_nCount) {
		memcpy(pDest, m_pValues, sizeof(double)*nMaxSize);
	}
	else {
		memcpy(pDest, m_pValues, sizeof(double)*m_nCount);
	}
}

void CKDNumbers::GetData(CKDTensor& dest) const {
	dest.SetData(m_pValues, m_nCount);
}

CKDUtilTokens::CKDUtilTokens(const char* pText) {
	m_nCount = 0;
	m_pBuffer = NULL;
	m_pValues = NULL;

	if (pText) SetData(pText);
}

CKDUtilTokens::~CKDUtilTokens() {
	delete[] m_pBuffer;
	delete[] m_pValues;
}


void CKDUtilTokens::SetData(const char* pText) {
	if (m_pValues || m_pBuffer) {
		delete[] m_pBuffer;
		delete[] m_pValues;

		m_nCount = 0;
		m_pBuffer = NULL;
		m_pValues = NULL;
	}

	int nLength = strlen(pText);
	m_pBuffer = new char[nLength + 1];
	const char** pValues = (const char**) new char*[nLength];
	int nCount = 0;

	strcpy_s(m_pBuffer, nLength + 1, pText);

	char* context;
	char* data = strtok_s(m_pBuffer, ", \n\r", &context);
	while (data) {
		pValues[nCount++] = data;
		data = strtok_s(NULL, ", \n\r", &context);
	}

	m_nCount = nCount;
	m_pValues = (const char**) new char*[m_nCount];
	memcpy(m_pValues, pValues, sizeof(const char*)*m_nCount);

	delete[] pValues;
}

const char* CKDUtilTokens::operator[](int idx) const {
	return m_pValues[idx];
}

FILE* CKDUtil::FileOpen(const char* path, const char* mode) {
	// create folder here...

	char buf[1024];
	_getcwd(buf, 1024);
	FILE* fid = NULL;
	if (fopen_s(&fid, path, mode) == 0) return fid;

	THROW_TEMP;
}

const char* CKDUtil::GetActFuncName(enum actFuncType func) {
	switch (func) {
	case act_func_bypass: return "bypass";
	case act_func_sigmoid:; return "sigmoid";
	case act_func_tanh:; return "tanh";
	case act_func_relu: return "relu";
	case act_func_softmax: return "softmax";
	};

	THROW_TEMP;

	return "unknown";
}

#ifdef _LINUX_

int fopen_s(FILE** fid, const char* fname, const char* mode) {
	*fid = fopen(fname, mode);
	return *fid ? 0 : 1;
}

int strcpy_s(char* pBuffer, int len, const char* pText) {
	return strcpy(pBuffer, pText) ? 0 : 1;
}

char* strtok_s(char* pBuffer, const char* pDelimeter, char** pContext) {
	return strtok(pBuffer, pDelimeter);
}

#endif