#pragma once

#include "../config/tinyxml2.h"
#include "../api/kodelldatatypes.h"

using namespace tinyxml2;
using namespace Kodell;

class CKDConfigManager {
public:
	CKDConfigManager();
	virtual ~CKDConfigManager();

	void Open(const char* filepath);
	void Close();
	
	XMLNode* GetRoot();

	int GetChildCount(XMLNode* node);
	XMLNode* GetParent(XMLNode* node);
	XMLNode* GetNthChild(XMLNode* node, int nth);
	void GetNodeInfo(XMLNode* node, ConfigNodeInfo* pInfo);

	XMLNode* LookupProject(const char* name);

protected:
	tinyxml2::XMLDocument m_configXML;
};