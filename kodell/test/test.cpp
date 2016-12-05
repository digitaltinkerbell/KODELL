#include "stdafx.h"

#include <unistd.h>
#include <stdio.h>
#include <string.h>

#include "../api/kodellapi.h"
#include "../api/kodelldatatypes.h"

using namespace Kodell;

char get_command(char** parg) {
	if (*parg && **parg) {
		char comm = **parg;
		(*parg)++;
		return comm;
	}
	else {
		return getchar();
	}
}
int main(int argc, char** argv)
{
	try {
		char* parg = (argc >= 2) ? argv[1] : NULL;
		ConfigManager configMan;

		char buf[1024];
		getcwd(buf, 1024);
		strcat(buf,"/TestData/config.xml");

		configMan.Open(buf);
		KHANDLE root = configMan.GetRoot();
		KHANDLE curr = root;

		ConfigNodeInfo nodeInfo;

		bool cont = true;

		while (cont) {
			if (parg == NULL || *parg == 0) {
				printf("\n");
			}
				
			int n = configMan.GetChildCount(curr);

			if (parg == NULL || *parg == 0) {
				if (curr != root) {
					printf("0: ..\n");
				}

				for (int i = 0; i < n; i++) {
					KHANDLE child = configMan.GetNthChild(curr, i);
					configMan.GetNodeInfo(child, &nodeInfo);
					printf("%d: %s %s\n", i+1, nodeInfo.type, nodeInfo.name);
				}

				printf("Q: quit\n");
				printf("Select Command: ");
			}

			while (1) {
				char ch = get_command(&parg);
				if (ch == '0' && curr != root) {
					curr = configMan.GetParent(curr);
					break;
				}
				if (ch == 'Q') {
					cont = false;
					break;
				}
				if (ch >= '1' && ch < '1' + n) {
					int k = ch - '1';
					KHANDLE child = configMan.GetNthChild(curr, k);
					configMan.GetNodeInfo(child, &nodeInfo);
					if (strcmp(nodeInfo.type, "folder") == 0) {
						curr = child;
						break;
					}
					else if (strcmp(nodeInfo.type, "project") == 0) {
						printf("Start Execute project '%s' here!!!\n", nodeInfo.name);
						
						Project project(&configMan, child);
						project.Execute();
						printf("End Execute project '%s' here!!!\n", nodeInfo.name);
						break;
					}
				}
			}
		}
		
		configMan.Close();
		getchar();
		return 0;
	}
	catch (CKDException ex) {
		printf("Exception in %s:%d\n", ex.m_pFile, ex.m_line);
	}
}
