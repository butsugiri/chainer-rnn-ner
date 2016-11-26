#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <map>
#include <stdio.h>
using namespace std;

stringstream ss;
#define cast(a,b) ss.clear(); ss<<a; ss>>b; ss.clear();


vector<map<string,string> > readfile(){
	vector< map<string,string> > seq;
	map<string,string> dic;
	string str,line;
	stringstream ss;
	while(getline(cin,line)){
		if(line=="") break;
		dic.clear();ss.clear();
		ss << line;
		ss>>str; dic["y"] = str; 
		ss>>str; dic["w"] = str; 
		ss>>str; dic["pos"] = str;
		ss>>str; dic["chk"] = str; 
		seq.push_back(dic);
	}
	return seq;
}

string apply_template(vector<map<string,string> >& seq, int id, vector<pair<string,int> >  temp){
	string ret,value,name,s;
	stringstream ss;
	ss<<temp[0].first<<"["<<temp[0].second<<"]";
	ss>>name;ss.clear();
	for(int i=1;i<temp.size();i++){
		ss<<"|"<<temp[i].first<<"["<<temp[i].second<<"]";
		ss>>s;name+=s;ss.clear();
	}
	
	string field; int offset,p;
	for(int i=0;i<temp.size();i++){
		field = temp[i].first; offset = temp[i].second;
		p = id + offset;
		if(p<0 or p>seq.size()-1) return "";
		value+=seq[p][field]; value+="|";
		
	}
	value=value.substr(0,value.size()-1);
	ret=name; ret+="="; ret+=value;
	return ret;
}


vector< vector<pair<string,int> > > templates;
void set_template(string str){
	string s; int x;
	vector<pair<string,int> > vec;
	s = str.substr(0, str.find(","));
	cast(str.substr(str.find(",")+1,str.size()-str.find(",")-1),x);
	vec.push_back(make_pair(s,x));
	templates.push_back(vec);
}

void set_template(string str1, string str2){
	string s; int x;
	vector<pair<string,int> > vec;
	s = str1.substr(0, str1.find(","));
	cast(str1.substr(str1.find(",")+1,str1.size()-str1.find(",")-1),x);
	vec.push_back(make_pair(s,x));
	
	s = str2.substr(0, str2.find(","));
	cast(str2.substr(str2.find(",")+1,str2.size()-str2.find(",")-1),x);
	vec.push_back(make_pair(s,x));
	templates.push_back(vec);
}

int main () {	
	//set template
	set_template("w,-2");
	set_template("w,-1");
	set_template("w,0");
	set_template("w,1");
	set_template("w,2");
	set_template("w,-2","w,-1");
	set_template("w,-1","w,0");
	set_template("w,0","w,1");
	set_template("w,1","w,2");
	set_template("pos,-2");
	set_template("pos,-1");
	set_template("pos,0");
	set_template("pos,1");
	set_template("pos,2");
	set_template("pos,-2","pos,-1");
	set_template("pos,-1","pos,0");
	set_template("pos,0","pos,1");
	set_template("pos,1","pos,2");
	set_template("chk,-2");
	set_template("chk,-1");
	set_template("chk,0");
	set_template("chk,1");
	set_template("chk,2");
	set_template("chk,-2","chk,-1");
	set_template("chk,-1","chk,0");
	set_template("chk,0","chk,1");
	set_template("chk,1","chk,2");
	set_template("iu,-2");
	set_template("iu,-1");
	set_template("iu,0");
	set_template("iu,1");
	set_template("iu,2");
	set_template("iu,-2","iu,-1");
	set_template("iu,-1","iu,0");
	set_template("iu,0","iu,1");
	set_template("iu,1","iu,2");
	
	//freopen("train","rt",stdin);
	//freopen("train.base","wt",stdout);
	vector<map<string,string> >seq;
	string attr; attr="";
	int cnt=0;
	while(1){
		cerr<<cnt++<<endl;
		seq = readfile(); if(seq.size()==0) break;
		for(int i=0;i<seq.size();i++){ 
			//Extract more characteristics of the input sequence
			seq[i]["iu"] = (seq[i]["w"]!="" and isupper(seq[i]["w"][0])) ? "True" : "False";
		}
		for(int i=0;i<seq.size();i++){
			cout<<seq[i]["y"];
			for(int j=0;j<templates.size();j++){
				attr = apply_template(seq, i, templates[j]);
				if(attr!=""){
					if(attr.find(":")!=-1) attr=attr.replace(attr.find(":"),1,"__COLON__");
					cout<<"\t"<<attr;
				}
			}
			cout<<endl;
		}
		cout<<endl;
	}
}




