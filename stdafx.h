#pragma once

#define _LINUX_

#define _getcwd getcwd

#define THROW_TEMP throw CKDException(__FILE__, __LINE__)

class CKDException {
public:
	CKDException(const char *pFile, int line) {
		m_pFile = pFile;
		m_line = line;
	}
	
	const char *m_pFile;
	int m_line;
};