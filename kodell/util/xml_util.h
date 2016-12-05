#pragma once

#include "../config/types.h"

#include <string>
using namespace std;

class CKDXmlUtil {
public:
	static XMLNode* SeekTagChild(XMLNode* parent, const char* tag);
	static XMLNode* SeekTagSibling(XMLNode* sibling, const char* tag);
	static XMLNode* SeekTagNamedChild(XMLNode* parent, const char* tag, const char* name);
	static XMLNode* GetNthChild(XMLNode* node, int nth);

	static const char* GetAttribute(XMLNode* node, const char* attr);
	static int GetIntAttr(XMLNode* node, const char* attr, int defVal=0);
	static int GetEnumAttr(XMLNode* node, const char* attr, const char* pNames[]);
	static int GetEnumTag(XMLNode* node, const char* pNames[]);
	static double GetDoubleAttr(XMLNode* node, const char* attr);

	static int GetChildCount(XMLNode* node);
	static int GetTagChildCount(XMLNode* node, const char* tag);

	static XMLNode* SeekDefClause(XMLNode* node, string name);

protected:
};
