import sys

isMCTest = False
isRACE = False
isDREAM = False

#Detemine input parameter
if len(sys.argv[1:])==0:
	#Run all modules
	isMCTest = True
	isRACE = True
	isDREAM = True
elif sys.argv[1] == '-MCTest' orsys.argv[2] == '-MCTest' or sys.argv[3] == '-MCTest':
    #Run MCTest Module
    isMCTest = True
elif sys.argv[1] == '-RACE' orsys.argv[2] == '-RACE' or sys.argv[3] == '-RACE':
    #Run RACE Module
    isRACE = True
elif sys.argv[1] == '-DREAM' orsys.argv[2] == '-DREAM' or sys.argv[3] == '-DREAM':
    #Run DREAM Module
    isMCTestDREAM = True
elif sys.argv[1] == '-Help':
    #Help of syntex
    print('usage: MC_QA.py -options')
    print('     : -MCTest Process MCTest dataset')
    print('     : -RACE   Process RACE dataset')
    print('     : -DREAM  Process DREAM dataset')
    sys.exit(2)
else:
    #Help of syntex
    print('usage: MC_QA.py -options')
    print('     : -MCTest Process MCTest dataset')
    print('     : -RACE   Process RACE dataset')
    print('     : -DREAM  Process DREAM dataset')
    sys.exit(2)

if isMCTest:
	##-------read training data---------


	##-------lemmatisation--------------'

	##-------Tokenization---------------'

	##-------Training-------------------'

	##-------QA Processing--------------'

if isRACE:
	##-------read training data---------


	##-------lemmatisation--------------'

	##-------Tokenization---------------'

	##-------Training-------------------'

	##-------QA Processing--------------'

if isDREAM:
	##-------read training data---------


	##-------lemmatisation--------------'

	##-------Tokenization---------------'

	##-------Training-------------------'

	##-------QA Processing--------------'

