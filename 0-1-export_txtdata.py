import argparse
import json, nltk, random, re

def main(solution, jsonfile):
    assert solution in ['none', 'tag', 'sub']
    
    print ("using human-named token solution '", solution, "' to build txt data file for OpenNMT.")
    
    with open(jsonfile) as f:
        valid_issues_train, valid_issues_val, valid_issues_test = json.load(f)
        
    valid_issues = valid_issues_train + valid_issues_val + valid_issues_test
    
    for idx, issue in enumerate(valid_issues):
        if idx % 50000 == 0:
            print ("current idx:", idx, "/", len(valid_issues))

        version_list = issue["_spctok"]["ver"]
        for version, stat in sorted(version_list.items(), key=lambda x:(len(x[0]))):
            if solution == 'tag': # verid40 & verid0 stands for tag before and after each human-named token, respectively
                issue['body'] = re.sub(re.escape(version), " verid40 " + version + " verid0 ", issue['body'], flags=re.IGNORECASE)
                issue['title'] = re.sub(re.escape(version), " " + version + " ", issue['title'], flags=re.IGNORECASE)
            elif solution == 'none': # just make it align with other solution
                issue['body'] = re.sub(re.escape(version), " " + version + " ", issue['body'], flags=re.IGNORECASE)
                issue['title'] = re.sub(re.escape(version), " " + version + " ", issue['title'], flags=re.IGNORECASE)
            elif solution == 'sub': # substitute with veridID, ID: appear order
                issue['body'] = re.sub(re.escape(version), " verid" + str(stat[0] + 1) + " ", issue['body'], flags=re.IGNORECASE)
                issue['title'] = re.sub(re.escape(version), " verid" + str(stat[0] + 1) + " ", issue['title'], flags=re.IGNORECASE)

        identifier_list = issue["_spctok"]["idt"]
        for idt, stat in sorted(identifier_list.items(), key=lambda x:(len(x[0]))):
            if solution == 'tag': # idtid40 & idtid0 stands for tag before and after each human-named token, respectively
                issue['body'] = re.sub(re.escape(idt), " idtid40 " + idt + " idtid0 ", issue['body'], flags=re.IGNORECASE)
                issue['title'] = re.sub(re.escape(idt), " " + idt + " ", issue['title'], flags=re.IGNORECASE)
            elif solution == 'none': # just make it align with other solution
                issue['body'] = re.sub(re.escape(idt), " " + idt + " ", issue['body'], flags=re.IGNORECASE)
                issue['title'] = re.sub(re.escape(idt), " " + idt + " ", issue['title'], flags=re.IGNORECASE)
            elif solution == 'sub': # substitute with idtidID, ID: appear order
                issue['body'] = re.sub(re.escape(idt), " idtid" + str(stat[0] + 1) + " ", issue['body'], flags=re.IGNORECASE)
                issue['title'] = re.sub(re.escape(idt), " idtid" + str(stat[0] + 1) + " ", issue['title'], flags=re.IGNORECASE)

        # final lowercase transformation & tokenize
        issue['body']  = " ".join(nltk.word_tokenize(issue['body'], preserve_line=False)).strip().lower()
        issue['title'] = " ".join(nltk.word_tokenize(issue['title'], preserve_line=False)).strip().lower()
    
    with open("body.train.txt", "w") as fbody, open("title.train.txt", "w") as ftitle:
        bodies = [x['body'] + "\n" for x in valid_issues_train]
        titles = [x['title'] + "\n" for x in valid_issues_train]
        fbody.writelines(bodies)
        ftitle.writelines(titles)
    with open("body.valid.txt", "w") as fbody, open("title.valid.txt", "w") as ftitle:
        bodies = [x['body'] + "\n" for x in valid_issues_val]
        titles = [x['title'] + "\n" for x in valid_issues_val]
        fbody.writelines(bodies)
        ftitle.writelines(titles)
    with open("body.test.txt", "w") as fbody, open("title.test.txt", "w") as ftitle:
        bodies = [x['body'] + "\n" for x in valid_issues_test]
        titles = [x['title'] + "\n" for x in valid_issues_test]
        fbody.writelines(bodies)
        ftitle.writelines(titles)
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare txt data file for OpenNMT')
    parser.add_argument('-solution', dest='solution', type=str, help='Any solution in tag, none, sub.', default='tag')
    parser.add_argument('-file', dest='jsonfile', default="refined_333563issues_reponobodytitlespctok.json",
                        help='JSON file of splitted dataset.')
    args = parser.parse_args()
    
    main(args.solution, args.jsonfile)