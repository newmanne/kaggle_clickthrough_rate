from datetime import datetime
from csv import DictReader
from math import exp, log, sqrt
from collections import Counter


# The goal of this file is to collect stats
# id,click,hour,C1,banner_pos,site_id,site_domain,site_category,app_id,app_domain,app_category,device_id,device_ip,device_model,device_type,device_conn_type,C14,C15,C16,C17,C18,C19,C20,C21

train = '/home/newmanne/cpsc540/train.csv'               # path to training file
statsfile = 'report.txt'
test = 'test.csv'                 # path to testing file

class ctr(object):
    def __init__(self):
        self.n_clicked = 0
        self.n = 0

    def process(self, row):
        self.n += 1
        if row['click'] == '1':
            self.n_clicked += 1

    def report(self):
        self.ctr = float(self.n_clicked) / self.n
        print "Out of %s entries, %s were clicked for a ctr of %s" % (self.n, self.n_clicked, self.ctr)

    def __str__(self):
        return "ctr"


class UniqueStats(object):
    def __init__(self):
        self.banner_pos = Counter()
        self.site_id = Counter()
        self.site_domain = Counter()
        self.site_category = Counter()
        self.app_id = Counter()
        self.app_domain = Counter()
        self.app_category = Counter()
        self.device_id = Counter()
        self.device_model = Counter()
        self.device_type = Counter()
        self.device_conn_type = Counter()
        self.c1 = Counter()
        self.c14 = Counter()
        self.c15 = Counter()
        self.c16 = Counter()
        self.c17 = Counter()
        self.c18 = Counter()
        self.c19 = Counter()
        self.c20 = Counter()
        self.c21 = Counter()
    
    def process(self, row):
        self.banner_pos.update([row['banner_pos']])
        self.site_id.update([row['site_id']])
        self.site_domain.update([row['site_domain']])
        self.site_category.update([row['site_category']]) 
        self.app_id.update([row['app_id']])
        self.app_domain.update([row['app_domain']])
        self.app_category.update([row['app_category']])
        self.device_id.update([row['device_id']])
        self.device_model.update([row['device_model']])
        self.device_type.update([row['device_type']])
        self.device_conn_type.update([row['device_conn_type']])
        self.c1.update([row['C1']])
        self.c14.update([row['C14']])
        self.c15.update([row['C15']])
        self.c16.update([row['C16']])
        self.c17.update([row['C17']])
        self.c18.update([row['C18']]) 
        self.c19.update([row['C19']]) 
        self.c20.update([row['C20']]) 
        self.c21.update([row['C21']]) 

    def report(self):
        with open(statsfile, 'w') as outfile:
            outfile.write('%s\n' % (self.banner_pos))
            outfile.write('%s\n' % (self.site_id))
            outfile.write('%s\n' % (self.site_domain))
            outfile.write('%s\n' % (self.site_category))
            outfile.write('%s\n' % (self.app_id))
            outfile.write('%s\n' % (self.app_domain))
            outfile.write('%s\n' % (self.app_category))
            outfile.write('%s\n' % (self.device_id))
            outfile.write('%s\n' % (self.device_model))
            outfile.write('%s\n' % (self.device_type))
            outfile.write('%s\n' % (self.device_conn_type))
            outfile.write('%s\n' % (self.c1))
            outfile.write('%s\n' % (self.c14))
            outfile.write('%s\n' % (self.c15))
            outfile.write('%s\n' % (self.c16))
            outfile.write('%s\n' % (self.c17))
            outfile.write('%s\n' % (self.c18))
            outfile.write('%s\n' % (self.c19))
            outfile.write('%s\n' % (self.c20))
            outfile.write('%s\n' % (self.c21))

    def __str__(self):
        return "UniqueStats"

def data(path):
    metrics = [ctr(), UniqueStats()]
    for t, row in enumerate(DictReader(open(path))):
        for metric in metrics:
            try:
                metric.process(row)
            except Exception as e:
                print "metric %s failed to process row %d" % (metric, t)
                print e               
    for metric in metrics:
        metric.report()

if __name__ == '__main__':
    start = datetime.now()
    data(train)
    elapsed_time = str(datetime.now() - start)
    print "Elapsed time %s" % elapsed_time