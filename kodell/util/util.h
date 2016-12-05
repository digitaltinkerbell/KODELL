#pragma once

#include <stdio.h>
#include "../util/xml_util.h"

enum actFuncType;

class CKDTensor;

class CKDNumbers {
public:
	CKDNumbers();
	CKDNumbers(const char* pText);
	virtual ~CKDNumbers();

	void SetData(const char* pText);
	void SetData(XMLNode* pNode);
	void GetData(double* pDest, int nMaxSize) const;
	void GetData(CKDTensor& dest) const;

protected:
	int m_nCount;
	double* m_pValues;
};

class CKDUtilTokens {
public:
	CKDUtilTokens(const char* pText);
	virtual ~CKDUtilTokens();

	void SetData(const char* pText);

	int size() const { return m_nCount; }
	const char* operator [](int idx) const;

protected:
	int m_nCount;
	char* m_pBuffer;
	const char** m_pValues;
};

class CKDUtil {
public:
	static FILE* FileOpen(const char* path, const char* mode);
	static const char* GetActFuncName(enum actFuncType func);
};

#ifdef _LINUX_

int fopen_s(FILE** fid, const char* fname, const char* mode);

#define sprintf_s snprintf
//int sprintf_s(char* buf, int len, const char* format, ...);

int strcpy_s(char* pBuffer, int len, const char* pText);

char* strtok_s(char* pBuffer, const char* pDelimeter, char** pContext);

#endif