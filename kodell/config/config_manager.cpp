#include "stdafx.h"

#include "../config/types.h"
#include "../config/config_manager.h"
#include "../util/xml_util.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

CKDConfigManager::CKDConfigManager() {
}

CKDConfigManager::~CKDConfigManager() {
}

void CKDConfigManager::Open(const char* filepath) {
	m_configXML.LoadFile(filepath);
}

void CKDConfigManager::Close() {
}

XMLNode* CKDConfigManager::GetRoot() {
	return CKDXmlUtil::SeekTagChild(m_configXML.RootElement(), "folder");
	/*
	XMLNode* root = m_configXML.RootElement();
	XMLNode *child;
	for (child = root->FirstChildElement(); child; child = child->NextSiblingElement()) {
		if (strcmp(child->Value(), ) == 0) return child;
	}
	return NULL;
	*/
}

int CKDConfigManager::GetChildCount(XMLNode* node) {
	return CKDXmlUtil::GetChildCount(node);
}

XMLNode* CKDConfigManager::GetParent(XMLNode* node) {
	return node->Parent();
}

XMLNode* CKDConfigManager::GetNthChild(XMLNode* node, int nth) {
	XMLNode *child = node->FirstChildElement();
	for (int i = 0; i < nth; i++) child = child->NextSiblingElement();
	return child;
}

void CKDConfigManager::GetNodeInfo(XMLNode* node, ConfigNodeInfo* pInfo) {
	pInfo->type = node->Value();
	pInfo->name = node->ToElement()->Attribute("name");
}

XMLNode* CKDConfigManager::LookupProject(const char* name) {
	XMLNode* projects = CKDXmlUtil::SeekTagChild(m_configXML.RootElement(), "projects");
	return CKDXmlUtil::SeekTagNamedChild(projects, "project", name);
}
