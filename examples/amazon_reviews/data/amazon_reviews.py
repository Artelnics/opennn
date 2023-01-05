'''
Artificial Intelligence Techniques SL	
artelnics@artelnics.com	

Your model has been exported to this python file.
You can manage it with the 'NeuralNetwork' class.	
Example:

	model = NeuralNetwork()	
	sample = [input_1, input_2, input_3, input_4, ...]	
	outputs = model.calculate_output(sample)

	Inputs Names: 	
	1 )phone
	2 )work
	3 )great
	4 )good
	5 )product
	6 )headset
	7 )qualiti
	8 )batteri
	9 )sound
	10 )use
	11 )one
	12 )well
	13 )ear
	14 )case
	15 )like
	16 )price
	17 )time
	18 )get
	19 )excel
	20 )us
	21 )ve
	22 )don
	23 )recommend
	24 )realli
	25 )comfort
	26 )problem
	27 )look
	28 )call
	29 )can
	30 )charg
	31 )servic
	32 )buy
	33 )make
	34 )best
	35 )fit
	36 )love
	37 )nice
	38 )also
	39 )charger
	40 )disappoint
	41 )purchas
	42 )just
	43 )item
	44 )new
	45 )ever
	46 )bluetooth
	47 )first
	48 )better
	49 )money
	50 )even
	51 )easi
	52 )car
	53 )tri
	54 )bought
	55 )year
	56 )wast
	57 )now
	58 )recept
	59 )doesn
	60 )thing
	61 )happi
	62 )poor
	63 )plug
	64 )drop
	65 )last
	66 )will
	67 )high
	68 )worst
	69 )cell
	70 )made
	71 )two
	72 )motorola
	73 )still
	74 )devic
	75 )bad
	76 )camera
	77 )design
	78 )piec
	79 )enough
	80 )fine
	81 )life
	82 )got
	83 )far
	84 )day
	85 )long
	86 )hear
	87 )clear
	88 )wear
	89 )volum
	90 )right
	91 )go
	92 )much
	93 )say
	94 )impress
	95 )hold
	96 )month
	97 )screen
	98 )pictur
	99 )think
	100 )week
	101 )turn
	102 )peopl
	103 )need
	104 )terribl
	105 )want
	106 )couldn
	107 )lot
	108 )light
	109 )pretti
	110 )button
	111 )expect
	112 )order
	113 )amazon
	114 )unit
	115 )cool
	116 )howev
	117 )low
	118 )end
	119 )everyth
	120 )custom
	121 )take
	122 )receiv
	123 )return
	124 )verizon
	125 )replac
	126 )connect
	127 )cheap
	128 )talk
	129 )horribl
	130 )jabra
	131 )found
	132 )without
	133 )hand
	134 )featur
	135 )never
	136 )back
	137 )sever
	138 )junk
	139 )feel
	140 )small
	141 )pair
	142 )quit
	143 )voic
	144 )broke
	145 )littl
	146 )complet
	147 )sinc
	148 )internet
	149 )ship
	150 )color
	151 )keep
	152 )perfect
	153 )break
	154 )didn
	155 )audio
	156 )compani
	157 )place
	158 )went
	159 )useless
	160 )seem
	161 )loud
	162 )real
	163 )stay
	164 )cabl
	165 )hour
	166 )softwar
	167 )way
	168 )nokia
	169 )start
	170 )pleas
	171 )signal
	172 )big
	173 )quick
	174 )earpiec
	175 )minut
	176 )find
	177 )within
	178 )hard
	179 )send
	180 )arriv
	181 )headphon
	182 )reason
	183 )difficult
	184 )black
	185 )anyon
	186 )came
	187 )clip
	188 )origin
	189 )come
	190 )crap
	191 )sturdi
	192 )protect
	193 )samsung
	194 )help
	195 )know
	196 )perform
	197 )around
	198 )contact
	199 )less
	200 )three
	201 )bar
	202 )simpl
	203 )side
	204 )put
	205 )everi
	206 )belt
	207 )definit
	208 )convers
	209 )differ
	210 )awesom
	211 )instruct
	212 )provid
	213 )fall
	214 )player
	215 )plan
	216 )data
	217 )import
	218 )strong
	219 )overal
	220 )line
	221 )sure
	222 )job
	223 )hate
	224 )die
	225 )set
	226 )drain
	227 )coupl
	228 )star
	229 )anyth
	230 )suck
	231 )valu
	232 )kind
	233 )mani
	234 )noth
	235 )size
	236 )easili
	237 )alway
	238 )part
	239 )clariti
	240 )old
	241 )charm
	242 )tool
	243 )anoth
	244 )razr
	245 )rang
	246 )fail
	247 )mobil
	248 )function
	249 )care
	250 )none
	251 )wife
	252 )plastic
	253 )free
	254 )especi
	255 )notic
	256 )packag
	257 )bargain
	258 )plantron
	259 )seller
	260 )fami
	261 )construct
	262 )earbud
	263 )avoid
	264 )must
	265 )ring
	266 )mic
	267 )flaw
	268 )allow
	269 )nois
	270 )obvious
	271 )leather
	272 )fast
	273 )decent
	274 )abl
	275 )goe
	276 )lightweight
	277 )mistak
	278 )worth
	279 )either
	280 )unreli
	281 )glad
	282 )said
	283 )extra
	284 )other
	285 )usb
	286 )treo
	287 )deal
	288 )satisfi
	289 )blue
	290 )kept
	291 )effect
	292 )support
	293 )rather
	294 )away
	295 )beep
	296 )rington
	297 )access
	298 )cingular
	299 )pocket
	300 )store
	301 )palm
	302 )unfortun
	303 )actual
	304 )keyboard
	305 )review
	306 )absolut
	307 )wrong
	308 )choic
	309 )lg
	310 )face
	311 )ago
	312 )weak
	313 )easier
	314 )later
	315 )lock
	316 )scratch
	317 )aw
	318 )left
	319 )thank
	320 )given
	321 )carri
	322 )network
	323 )touch
	324 )tone
	325 )handsfre
	326 )commun
	327 )refund
	328 )coverag
	329 )sharp
	330 )outlet
	331 )holster
	332 )happen
	333 )iphon
	334 )maintain
	335 )cut
	336 )forev
	337 )pay
	338 )descript
	339 )current
	340 )normal
	341 )plus
	342 )state
	343 )sprint
	344 )speaker
	345 )tinni
	346 )despit
	347 )accept
	348 )wire
	349 )stop
	350 )messag
	351 )fantast
	352 )video
	353 )area
	354 )download
	355 )appear
	356 )experi
	357 )final
	358 )mess
	359 )soni
	360 )market
	361 )oper
	362 )addit
	363 )push
	364 )happier
	365 )probabl
	366 )save
	367 )secur
	368 )give
	369 )caus
	370 )wireless
	371 )flawless
	372 )practic
	373 )flip
	374 )unaccept
	375 )uncomfort
	376 )lack
	377 )microphon
	378 )jawbon
	379 )instal
	380 )ll
	381 )slow
	382 )describ
	383 )cellphon
	384 )result
	385 )exchang
	386 )incred
	387 )lost
	388 )poorli
	389 )extrem
	390 )friend
	391 )understand
	392 )comput
	393 )amaz
	394 )read
	395 )simp
	396 )expens
	397 )thought
	398 )wasn
	399 )defect
	400 )might
	401 )unless
	402 )form
	403 )oh
	404 )dead
	405 )keypad
	406 )beauti
	407 )home
	408 )everyon
	409 )setup
	410 )yet
	411 )run
	412 )display
	413 )perhap
	414 )serious
	415 )may
	416 )though
	417 )includ
	418 )ipod
	419 )troubl
	420 )number
	421 )basic
	422 )match
	423 )book
	424 )pc
	425 )worthless
	426 )let
	427 )own
	428 )cover
	429 )front
	430 )that
	431 )rock
	432 )bt
	433 )eas
	434 )second
	435 )cost
	436 )almost
	437 )super
	438 )instead
	439 )decis
	440 )rate
	441 )longer
	442 )buyer
	443 )direct
	444 )static
	445 )play
	446 )listen
	447 )complaint
	448 )dont
	449 )websit
	450 )particular
	451 )skype
	452 )mp3
	453 )clearli
	454 )mention
	455 )alon
	456 )moto
	457 )figur
	458 )parti
	459 )exact
	460 )forget
	461 )warn
	462 )wonder
	463 )tooth
	464 )wait
	465 )stupid
	466 )advis
	467 )sometim
	468 )model
	469 )att
	470 )tech
	471 )note
	472 )essenti
	473 )whole
	474 )person
	475 )dock
	476 )station
	477 )d807
	478 )complain
	479 )advertis
	480 )wise
	481 )handi
	482 )ador
	483 )compar
	484 )revers
	485 )cheaper
	486 )flimsi
	487 )smell
	488 )music
	489 )purpos
	490 )whatsoev
	491 )gel
	492 )rip
	493 )along
	494 )improv
	495 )cradl
	496 )activ
	497 )sudden
	498 )recharg
	499 )glass
	500 )switch
	501 )auto
	502 )game
	503 )neither
	504 )seri
	505 )cant
	506 )dial
	507 )driv
	508 )gotten
	509 )quiet
	510 )spring
	511 )power
	512 )unus
	513 )open
	514 )except
	515 )earphon
	516 )crisp
	517 )chines
	518 )flash
	519 )third
	520 )period
	521 )wall
	522 )loop
	523 )utter
	524 )someth
	525 )warranti
	526 )told
	527 )wind
	528 )echo
	529 )pull
	530 )red
	531 )reach
	532 )wow
	533 )control
	534 )rest
	535 )loos
	536 )ok
	537 )lose
	538 )mak
	539 )insid
	540 )laptop
	541 )answer
	542 )brand
	543 )feet
	544 )togeth
	545 )manag
	546 )bottom
	547 )date
	548 )protector
	549 )next
	550 )show
	551 )etc
	552 )regret
	553 )sourc
	554 )timefram
	555 )catch
	556 )regard
	557 )cancel
	558 )im
	559 )fire
	560 )abil
	561 )pda
	562 )user
	563 )slid
	564 )won
	565 )worthwhil
	566 )stuff
	567 )cumbersom
	568 )procedur
	569 )sunglass
	570 )gadget
	571 )larg
	572 )pad
	573 )joke
	574 )dissapoint
	575 )apart
	576 )beat
	577 )least
	578 )pain
	579 )com
	580 )broken
	581 )re
	582 )adapt
	583 )forc
	584 )key
	585 )bare
	586 )continu
	587 )total
	588 )numer
	589 )eargel
	590 )took
	591 )window
	592 )accident
	593 )refus
	594 )bewar
	595 )previous
	596 )ask
	597 )white
	598 )unhappi
	599 )tick
	600 )embarrass
	601 )tip
	602 )igo
	603 )liter
	604 )huge
	605 )crack
	606 )consum
	607 )blackberri
	608 )background
	609 )w810i
	610 )superb
	611 )certainli
	612 )link
	613 )usual
	614 )thin
	615 )sex
	616 )nearli
	617 )dozen
	618 )wouldn
	619 )experienc
	620 )pros
	621 )tremend
	622 )technolog
	623 )bother
	624 )recognit
	625 )hous
	626 )slim
	627 )storag
	628 )buzz
	629 )overrid
	630 )option
	631 )offer
	632 )room
	633 )issu
	634 )felt
	635 )full
	636 )although
	637 )sleek
	638 )excit
	639 )check
	640 )extend
	641 )major
	642 )graphic
	643 )done
	644 )logitech
	645 )hop
	646 )resolut
	647 )tell
	648 )bit
	649 )ergonom
	650 )sign
	651 )sketchi
	652 )smallest
	653 )biggest
	654 )mark
	655 )superfast
	656 )explain
	657 )pass
	658 )theori
	659 )stand
	660 )funni
	661 )freedom
	662 )occupi
	663 )readi
	664 )sent
	665 )copier
	666 )shape
	667 )randomli
	668 )truli
	669 )worn
	670 )ringer
	671 )freeway
	672 )balanc
	673 )ca
	674 )prime
	675 )whose
	676 )upbeat
	677 )tight
	678 )forgeri
	679 )abound
	680 )span
	681 )soft
	682 )jack
	683 )crash
	684 )reboot
	685 )curv
	686 )startac
	687 )outperform
	688 )whistl
	689 )china
	690 )v325i
	691 )sim
	692 )3o
	693 )mediocr
	694 )bell
	695 )mov
	696 )4s
	697 )multipl
	698 )elsewher
	699 )iam
	700 )sensit
	701 )imac
	702 )extern
	703 )slip
	704 )tv
	705 )whoa
	706 )exclaim
	707 )distract
	708 )entir
	709 )simpler
	710 )cbr
	711 )mp3s
	712 )prefer
	713 )cord
	714 )prevent
	715 )grip
	716 )slide
	717 )via
	718 )media
	719 )shot
	720 )sos
	721 )mini
	722 )near
	723 )whine
	724 )pen
	725 )pack
	726 )buyit
	727 )finger
	728 )highi
	729 )believ
	730 )steep
	731 )point
	732 )juic
	733 )qwerti
	734 )haul
	735 )replacementr
	736 )discard
	737 )post
	738 )detail
	739 )comment
	740 )grey
	741 )hundr
	742 )ac
	743 )guess
	744 )exist
	745 )cds
	746 )surpris
	747 )crappi
	748 )infuri
	749 )walkman
	750 )europ
	751 )asia
	752 )deffinit
	753 )cent
	754 )monkey
	755 )beh
	756 )stylish
	757 )fraction
	758 )min
	759 )fabul
	760 )e715
	761 )seeen
	762 )interfac
	763 )decad
	764 )compet
	765 )compromis
	766 )700w
	767 )couldnt
	768 )transceiv
	769 )steer
	770 )genuin
	771 )tungsten
	772 )heavi
	773 )signific
	774 )promis
	775 )brows
	776 )tini
	777 )four
	778 )web
	779 )latch
	780 )visor
	781 )address
	782 )supertooth
	783 )snug
	784 )e2
	785 )flipphon
	786 )smooth
	787 )studi
	788 )interest
	789 )sin
	790 )industri
	791 )track
	792 )detach
	793 )winner
	794 )somehow
	795 )jiggl
	796 )contract
	797 )shooter
	798 )delay
	799 )bitpim
	800 )program
	801 )transfer
	802 )accessori
	803 )manufactur
	804 )muffl
	805 )speed
	806 )incom
	807 )upload
	808 )unlik
	809 )resist
	810 )over
	811 )build
	812 )produc
	813 )receipt
	814 )luck
	815 )linksi
	816 )refurb
	817 )faster
	818 )leaf
	819 )land
	820 )materi
	821 )offici
	822 )oem
	823 )loudest
	824 )competitor
	825 )alot
	826 )definitli
	827 )unintellig
	828 )word
	829 )restart
	830 )bend
	831 )cutout
	832 )metal
	833 )stress
	834 )leopard
	835 )print
	836 )wild
	837 )saggi
	838 )floppi
	839 )snap
	840 )fliptop
	841 )wobbl
	842 )eventu
	843 )sister
	844 )cassett
	845 )dirti
	846 )haven
	847 )sensor
	848 )reliabl
	849 )breakag
	850 )ir
	851 )counterfeit
	852 )see
	853 )travl
	854 )swivel
	855 )seat
	856 )dual
	857 )damag
	858 )bottowm
	859 )gimmick
	860 )top
	861 )discomfort
	862 )trust
	863 )basem
	864 )confus
	865 )upstair
	866 )holder
	867 )negativeli
	868 )girl
	869 )wake
	870 )styl
	871 )restock
	872 )fee
	873 )darn
	874 )lousi
	875 )seen
	876 )sweetest
	877 )hook
	878 )canal
	879 )unsatisfactori
	880 )certain
	881 )hype
	882 )assum
	883 )lens
	884 )text
	885 )tricki
	886 )blew
	887 )flop
	888 )smudg
	889 )infra
	890 )port
	891 )irda
	892 )s710a
	893 )fulfil
	894 )requir
	895 )fact
	896 )keen
	897 )lap
	898 )peachi
	899 )mine
	900 )christma
	901 )otherwis
	902 )joy
	903 )satisif
	904 )speakerphon
	905 )spec
	906 )armband
	907 )allot
	908 )clearer
	909 )ericson
	910 )z500a
	911 )motor
	912 )center
	913 )voltag
	914 )hum
	915 )equip
	916 )son
	917 )hey
	918 )pleasant
	919 )supris
	920 )dustpan
	921 )indoor
	922 )dispos
	923 )puff
	924 )smoke
	925 )conveni
	926 )ride
	927 )smoother
	928 )nano
	929 )ant
	930 )reccommend
	931 )highest
	932 )anti
	933 )glare
	934 )relat
	935 )reccomend
	936 )smartphon
	937 )wont
	938 )atleast
	939 )amp
	940 )reoccur
	941 )blueant
	942 )sold
	943 )classi
	944 )krussel
	945 )tracfonewebsit
	946 )toactiv
	947 )texa
	948 )dit
	949 )mainli
	950 )soon
	951 )whatev
	952 )disapoin
	953 )somewher
	954 )share
	955 )metro
	956 )pcs
	957 )sch
	958 )r450
	959 )slider
	960 )premium
	961 )plenti
	962 )capac
	963 )confort
	964 )somewhat
	965 )brilliant
	966 )ps3
	967 )five
	968 )cheapi
	969 )shout
	970 )telephon
	971 )yes
	972 )shini
	973 )grtting
	974 )v3c
	975 )thumb
	976 )accompani
	977 )exceed
	978 )instanc
	979 )sight
	980 )improp
	981 )everywher
	982 )awkward
	983 )hope
	984 )father
	985 )v265
	986 )ideal
	987 )intermitt
	988 )row
	989 )nightmar
	990 )discount
	991 )els
	992 )creak
	993 )wooden
	994 )floor
	995 )gener
	996 )inconspicu
	997 )boot
	998 )slowli
	999 )sorri
	1000 )imposs
	1001 )upgrad
	1002 )anywher
	1003 )securli
	1004 )possibl
	1005 )doubl
	1006 )entertain
	1007 )realiz
	1008 )activesync
	1009 )optim
	1010 )synchron
	1011 )disgust
	1012 )coupon
	1013 )rare
	1014 )finish
	1015 )good7
	1016 )leak
	1017 )hot
	1018 )accord
	1019 )applifi
	1020 )paus
	1021 )special
	1022 )style
	1023 )transmiss
	1024 )s11
	1025 )pixel
	1026 )appeal
	1027 )mega
	1028 )drivng
	1029 )h500
	1030 )transmitt
	1031 )tape
	1032 )embarass
	1033 )hurt
	1034 )fm
	1035 )averag
	1036 )drawback
	1037 )avail
	1038 )capabl
	1039 )frequenty
	1040 )odd
	1041 )adhes
	1042 )inexpens
	1043 )add
	1044 )boost
	1045 )concret
	1046 )knock
	1047 )wood
	1048 )transform
	1049 )organiz
	1050 )soyo
	1051 )transmit
	1052 )sit
	1053 )vehicl
	1054 )song
	1055 )jerk
	1056 )los
	1057 )angel
	1058 )starter
	1059 )skip
	1060 )loudspeak
	1061 )bumper
	1062 )weird
	1063 )era
	1064 )hoursth
	1065 )thereplac
	1066 )cheapli
	1067 )jabra350
	1068 )distort
	1069 )yell
	1070 )fun
	1071 )microsoft
	1072 )weight
	1073 )forgot
	1074 )nicer
	1075 )iriv
	1076 )spinn
	1077 )hit
	1078 )fond
	1079 )magnet
	1080 )strap
	1081 )psych
	1082 )appoint
	1083 )fool
	1084 )wit
	1085 )click
	1086 )encourag
	1087 )self
	1088 )portrait
	1089 )outsid
	1090 )exterior
	1091 )electron
	1092 )magic
	1093 )angl
	1094 )promptli
	1095 )strang
	1096 )today
	1097 )invest
	1098 )dollar
	1099 )prettier
	1100 )deaf
	1101 )correct
	1102 )reciev
	1103 )prompt
	1104 )eleg
	1105 )owner
	1106 )pleather
	1107 )kit
	1108 )facepl
	1109 )cingulair
	1110 )headband
	1111 )bud
	1112 )scari
	1113 )greater
	1114 )stereo
	1115 )ft
	1116 )absolutel
	1117 )hair
	1118 )integr
	1119 )load
	1120 )potenti
	1121 )fri
	1122 )unbear
	1123 )giv
	1124 )gave
	1125 )boy
	1126 )rotat
	1127 )excess
	1128 )intend
	1129 )everyday
	1130 )seamless
	1131 )contstruct
	1132 )hing
	1133 )overnit
	1134 )shine
	1135 )sync
	1136 )tie
	1137 )waaay
	1138 )cute
	1139 )defeat
	1140 )seper
	1141 )penni
	1142 )wallet
	1143 )type
	1144 )excruti
	1145 )mere
	1146 )thru
	1147 )aspect
	1148 )glove
	1149 )authent
	1150 )durabl
	1151 )gosh
	1152 )attract
	1153 )favorit
	1154 )factor
	1155 )rubber
	1156 )petroleum
	1157 )ericsson
	1158 )suppos
	1159 )clock
	1160 )remov
	1161 )antena
	1162 )liv
	1163 )complim
	1164 )fairli
	1165 )usag
	1166 )roam
	1167 )immedi
	1168 )ngage
	1169 )needless
	1170 )alarm
	1171 )mean
	1172 )bmw
	1173 )convert
	1174 )riington
	1175 )earbug
	1176 )anyway
	1177 )situat
	1178 )onlin
	1179 )appar
	1180 )lesson
	1181 )learn
	1182 )invent
	1183 )handset
	1184 )cat
	1185 )attack
	1186 )strip
	1187 )destroy
	1188 )garbl
	1189 )razor
	1190 )v3i
	1191 )child
	1192 )someon
	1193 )shouldv
	1194 )bland
	1195 )sooner
	1196 )engin
	1197 )clever
	1198 )mostli
	1199 )frequently4
	1200 )freez
	1201 )infatu
	1202 )flush
	1203 )tracfon
	1204 )manual
	1205 )toilet
	1206 )channel
	1207 )threw
	1208 )browser
	1209 )inch
	1210 )kitchen
	1211 )counter
	1212 )laugh
	1213 )trunk
	1214 )mother
	1215 )hitch
	1216 )ampl
	1217 )file
	1218 )garbag
	1219 )increas
	1220 )proper
	1221 )miss
	1222 )shift
	1223 )bubbl
	1224 )peel
	1225 )droid
	1226 )zero
	1227 )exercis
	1228 )frustrat
	1229 )earset
	1230 )geeki
	1231 )earlier
	1232 )os
	1233 )latest
	1234 )frog
	1235 )eye
	1236 )whether
	1237 )aluminum
	1238 )argu
	1239 )vx
	1240 )handheld
	1241 )hs850
	1242 )outgo
	1243 )waterproof
	1244 )standard
	1245 )edg
	1246 )pant
	1247 )gonna
	1248 )ugli
	1249 )shield
	1250 )incredi
	1251 )toast
	1252 )mind
	1253 )gentl
	1254 )navig
	1255 )due
	1256 )human
	1257 )copi
	1258 )dna
	1259 )walk
	1260 )calendar
	1261 )wip
	1262 )strength
	1263 )louder
	1264 )short
	1265 )menus
	1266 )bougth
	1267 )recess
	1268 )smok
	1269 )shouldn
	1270 )effort
	1271 )posses
	1272 )idea
	1273 )trash
	1274 )research
	1275 )develop
	1276 )divis
	1277 )killer
	1278 )glu
	1279 )ooz
	1280 )patient
	1281 )wiref
	1282 )combin
	1283 )inform
	1284 )aggrav
	1285 )enjoy
	1286 )virgin
	1287 )muddi
	1288 )cas
	1289 )insert
	1290 )ad
	1291 )isn
	1292 )plantroninc
	1293 )mode
	1294 )disapoint
	1295 )fourth
	1296 )fix
	1297 )pic
	1298 )2mp
	1299 )embed
	1300 )constant
	1301 )l7c
	1302 )wi
	1303 )search
	1304 )mail
	1305 )buck
	1306 )lit
	1307 )follow
	1308 )relativeli
	1309 )portabl
	1310 )colleagu
	1311 )disconnect
	1312 )fulli
	1313 )bed
	1314 )night
	1315 )fi
	1316 )morn
	1317 )memori
	1318 )card
	1319 )neat
	1320 )hat
	1321 )recent
	1322 )shipment
	1323 )solid
	1324 )surefir
	1325 )gx2
	1326 )awsom
	1327 )sanyo
	1328 )surviv
	1329 )blacktop
	1330 )ill
	1331 )late
	1332 )enter
	1333 )modest
	1334 )cellular
	1335 )megapixel
	1336 )render
	1337 )wish
	1338 )bt50
	1339 )earpad
	1340 )mechan
	1341 )displeas
	1342 )submerg
	1343 )risk
	1344 )built
	1345 )imag
	1346 )stream
	1347 )restor
	1348 )jx
	1349 )backlight
	1350 )role
	1351 )recogn
	1352 )piti
	1353 )respect
	1354 )usabl
	1355 )bulki
	1356 )stuck
	1357 )max
	1358 )mute
	1359 )hybrid
	1360 )palmtop
	1361 )kindl
	1362 )world
	1363 )bt250v
	1364 )crawl
	1365 )imagin
	1366 )bose
	1367 )15g
	1368 )commerci
	1369 )nyc
	1370 )commut
	1371 )mislead
	1372 )v1
	1373 )photo
	1374 )vx9900
	1375 )abhor
	1376 )remors
	1377 )accessoryon
	1378 )inexcus
	1379 )chang
	1380 )carrier
	1381 )tmobil
	1382 )updat
	1383 )thorn
	1384 )purcash
	1385 )deliveri
	1386 )cours
	1387 )env
	1388 )rocket
	1389 )destin
	1390 )unknown
	1391 )condit
	1392 )bluetoooth
	1393 )bill
	1394 )machin
	1395 )pric
	1396 )overnight

You can predict with a batch of samples using calculate_batch_output method	
IMPORTANT: input batch must be <class 'numpy.ndarray'> type	
Example_1:	
	model = NeuralNetwork()	
	input_batch = np.array([[1, 2], [4, 5]], np.int32)	
	outputs = model.calculate_batch_output(input_batch)
Example_2:	
	input_batch = pd.DataFrame( {'col1': [1, 2], 'col2': [3, 4]})	
	outputs = model.calculate_batch_output(input_batch.values)
'''

import numpy as np

class NeuralNetwork:
 
	def __init__(self):
 
		self.parameters_number = 8389
 
	def scaling_layer(self,inputs):

		outputs = [None] * 1396

		outputs[0] = (inputs[0]-0.1780000031)/0.4104686081
		outputs[1] = (inputs[1]-0.1120000035)/0.3279688358
		outputs[2] = (inputs[2]-0.09899999946)/0.3054378629
		outputs[3] = (inputs[3]-0.07699999958)/0.2777555585
		outputs[4] = (inputs[4]-0.05600000173)/0.2300367802
		outputs[5] = (inputs[5]-0.0549999997)/0.2367087454
		outputs[6] = (inputs[6]-0.04899999872)/0.2159760296
		outputs[7] = (inputs[7]-0.04800000042)/0.2185034156
		outputs[8] = (inputs[8]-0.04699999839)/0.2298778892
		outputs[9] = (inputs[9]-0.04300000146)/0.2078321278
		outputs[10] = (inputs[10]-0.04199999943)/0.2151331753
		outputs[11] = (inputs[11]-0.04199999943)/0.2056168169
		outputs[12] = (inputs[12]-0.04100000113)/0.2266503274
		outputs[13] = (inputs[13]-0.03400000185)/0.1813198179
		outputs[14] = (inputs[14]-0.03299999982)/0.1842415333
		outputs[15] = (inputs[15]-0.03299999982)/0.1787258834
		outputs[16] = (inputs[16]-0.03200000152)/0.1760880649
		outputs[17] = (inputs[17]-0.03200000152)/0.1871123016
		outputs[18] = (inputs[18]-0.02899999917)/0.1678903997
		outputs[19] = (inputs[19]-0.02800000086)/0.1710124165
		outputs[20] = (inputs[20]-0.02800000086)/0.1823437661
		outputs[21] = (inputs[21]-0.02800000086)/0.1767689139
		outputs[22] = (inputs[22]-0.0270000007)/0.1621644199
		outputs[23] = (inputs[23]-0.02600000054)/0.1592147946
		outputs[24] = (inputs[24]-0.02500000037)/0.162485078
		outputs[25] = (inputs[25]-0.02500000037)/0.1562030762
		outputs[26] = (inputs[26]-0.02500000037)/0.1685330868
		outputs[27] = (inputs[27]-0.02400000021)/0.1595288366
		outputs[28] = (inputs[28]-0.02400000021)/0.153125599
		outputs[29] = (inputs[29]-0.02300000004)/0.156510368
		outputs[30] = (inputs[30]-0.02300000004)/0.1499783099
		outputs[31] = (inputs[31]-0.02300000004)/0.1499783099
		outputs[32] = (inputs[32]-0.02300000004)/0.1499783099
		outputs[33] = (inputs[33]-0.02300000004)/0.156510368
		outputs[34] = (inputs[34]-0.02300000004)/0.1499783099
		outputs[35] = (inputs[35]-0.02300000004)/0.1499783099
		outputs[36] = (inputs[36]-0.02300000004)/0.1499783099
		outputs[37] = (inputs[37]-0.02199999988)/0.1467567235
		outputs[38] = (inputs[38]-0.02199999988)/0.1659624726
		outputs[39] = (inputs[39]-0.02099999972)/0.1434558481
		outputs[40] = (inputs[40]-0.02099999972)/0.1434558481
		outputs[41] = (inputs[41]-0.02099999972)/0.1502716988
		outputs[42] = (inputs[42]-0.01999999955)/0.140070051
		outputs[43] = (inputs[43]-0.01999999955)/0.1470429301
		outputs[44] = (inputs[44]-0.01899999939)/0.1365930438
		outputs[45] = (inputs[45]-0.01899999939)/0.1365930438
		outputs[46] = (inputs[46]-0.01899999939)/0.1365930438
		outputs[47] = (inputs[47]-0.01899999939)/0.1437346786
		outputs[48] = (inputs[48]-0.01899999939)/0.1365930438
		outputs[49] = (inputs[49]-0.01799999923)/0.1330176443
		outputs[50] = (inputs[50]-0.01799999923)/0.1330176443
		outputs[51] = (inputs[51]-0.01799999923)/0.1403413564
		outputs[52] = (inputs[52]-0.01700000092)/0.1293357164
		outputs[53] = (inputs[53]-0.01700000092)/0.1368566006
		outputs[54] = (inputs[54]-0.01700000092)/0.1293357164
		outputs[55] = (inputs[55]-0.01700000092)/0.1293357164
		outputs[56] = (inputs[56]-0.01700000092)/0.1293357164
		outputs[57] = (inputs[57]-0.01700000092)/0.1293357164
		outputs[58] = (inputs[58]-0.01600000076)/0.1255378872
		outputs[59] = (inputs[59]-0.01499999966)/0.121613279
		outputs[60] = (inputs[60]-0.01499999966)/0.121613279
		outputs[61] = (inputs[61]-0.01499999966)/0.1295831501
		outputs[62] = (inputs[62]-0.01499999966)/0.121613279
		outputs[63] = (inputs[63]-0.01499999966)/0.1295831501
		outputs[64] = (inputs[64]-0.01499999966)/0.1295831501
		outputs[65] = (inputs[65]-0.01499999966)/0.121613279
		outputs[66] = (inputs[66]-0.01400000043)/0.1175492108
		outputs[67] = (inputs[67]-0.01400000043)/0.125776872
		outputs[68] = (inputs[68]-0.01400000043)/0.1175492108
		outputs[69] = (inputs[69]-0.01400000043)/0.1175492108
		outputs[70] = (inputs[70]-0.01400000043)/0.1175492108
		outputs[71] = (inputs[71]-0.01400000043)/0.1175492108
		outputs[72] = (inputs[72]-0.01400000043)/0.1175492108
		outputs[73] = (inputs[73]-0.01400000043)/0.1175492108
		outputs[74] = (inputs[74]-0.01400000043)/0.1175492108
		outputs[75] = (inputs[75]-0.01300000027)/0.1218435317
		outputs[76] = (inputs[76]-0.01300000027)/0.1133306846
		outputs[77] = (inputs[77]-0.01300000027)/0.1133306846
		outputs[78] = (inputs[78]-0.01300000027)/0.1133306846
		outputs[79] = (inputs[79]-0.01300000027)/0.1133306846
		outputs[80] = (inputs[80]-0.01300000027)/0.1133306846
		outputs[81] = (inputs[81]-0.01300000027)/0.1133306846
		outputs[82] = (inputs[82]-0.01300000027)/0.1133306846
		outputs[83] = (inputs[83]-0.01300000027)/0.1133306846
		outputs[84] = (inputs[84]-0.01300000027)/0.1297992617
		outputs[85] = (inputs[85]-0.01300000027)/0.1133306846
		outputs[86] = (inputs[86]-0.0120000001)/0.1089397445
		outputs[87] = (inputs[87]-0.0120000001)/0.1089397445
		outputs[88] = (inputs[88]-0.0120000001)/0.1089397445
		outputs[89] = (inputs[89]-0.0120000001)/0.1089397445
		outputs[90] = (inputs[90]-0.0120000001)/0.1089397445
		outputs[91] = (inputs[91]-0.0120000001)/0.1089397445
		outputs[92] = (inputs[92]-0.01099999994)/0.1043546349
		outputs[93] = (inputs[93]-0.01099999994)/0.1043546349
		outputs[94] = (inputs[94]-0.01099999994)/0.1043546349
		outputs[95] = (inputs[95]-0.01099999994)/0.1043546349
		outputs[96] = (inputs[96]-0.01099999994)/0.1043546349
		outputs[97] = (inputs[97]-0.01099999994)/0.1135424674
		outputs[98] = (inputs[98]-0.009999999776)/0.09954853356
		outputs[99] = (inputs[99]-0.009999999776)/0.09954853356
		outputs[100] = (inputs[100]-0.009999999776)/0.09954853356
		outputs[101] = (inputs[101]-0.009999999776)/0.09954853356
		outputs[102] = (inputs[102]-0.009999999776)/0.09954853356
		outputs[103] = (inputs[103]-0.009999999776)/0.09954853356
		outputs[104] = (inputs[104]-0.009999999776)/0.09954853356
		outputs[105] = (inputs[105]-0.009999999776)/0.09954853356
		outputs[106] = (inputs[106]-0.009999999776)/0.09954853356
		outputs[107] = (inputs[107]-0.009999999776)/0.1091417074
		outputs[108] = (inputs[108]-0.009999999776)/0.09954853356
		outputs[109] = (inputs[109]-0.009999999776)/0.09954853356
		outputs[110] = (inputs[110]-0.009999999776)/0.09954853356
		outputs[111] = (inputs[111]-0.008999999613)/0.1045463085
		outputs[112] = (inputs[112]-0.008999999613)/0.09448771179
		outputs[113] = (inputs[113]-0.008999999613)/0.09448771179
		outputs[114] = (inputs[114]-0.008999999613)/0.09448771179
		outputs[115] = (inputs[115]-0.008999999613)/0.09448771179
		outputs[116] = (inputs[116]-0.008999999613)/0.09448771179
		outputs[117] = (inputs[117]-0.008999999613)/0.09448771179
		outputs[118] = (inputs[118]-0.008999999613)/0.09448771179
		outputs[119] = (inputs[119]-0.008999999613)/0.09448771179
		outputs[120] = (inputs[120]-0.008999999613)/0.09448771179
		outputs[121] = (inputs[121]-0.008999999613)/0.09448771179
		outputs[122] = (inputs[122]-0.008999999613)/0.09448771179
		outputs[123] = (inputs[123]-0.008999999613)/0.09448771179
		outputs[124] = (inputs[124]-0.008999999613)/0.09448771179
		outputs[125] = (inputs[125]-0.008999999613)/0.09448771179
		outputs[126] = (inputs[126]-0.008999999613)/0.1045463085
		outputs[127] = (inputs[127]-0.008999999613)/0.1045463085
		outputs[128] = (inputs[128]-0.00800000038)/0.0997293666
		outputs[129] = (inputs[129]-0.00800000038)/0.08912880719
		outputs[130] = (inputs[130]-0.00800000038)/0.08912880719
		outputs[131] = (inputs[131]-0.00800000038)/0.08912880719
		outputs[132] = (inputs[132]-0.00800000038)/0.08912880719
		outputs[133] = (inputs[133]-0.00800000038)/0.08912880719
		outputs[134] = (inputs[134]-0.00800000038)/0.08912880719
		outputs[135] = (inputs[135]-0.00800000038)/0.08912880719
		outputs[136] = (inputs[136]-0.00800000038)/0.0997293666
		outputs[137] = (inputs[137]-0.00800000038)/0.08912880719
		outputs[138] = (inputs[138]-0.00800000038)/0.08912880719
		outputs[139] = (inputs[139]-0.00800000038)/0.08912880719
		outputs[140] = (inputs[140]-0.00800000038)/0.08912880719
		outputs[141] = (inputs[141]-0.00800000038)/0.08912880719
		outputs[142] = (inputs[142]-0.00800000038)/0.08912880719
		outputs[143] = (inputs[143]-0.00800000038)/0.08912880719
		outputs[144] = (inputs[144]-0.00800000038)/0.08912880719
		outputs[145] = (inputs[145]-0.007000000216)/0.08341437578
		outputs[146] = (inputs[146]-0.007000000216)/0.08341437578
		outputs[147] = (inputs[147]-0.007000000216)/0.08341437578
		outputs[148] = (inputs[148]-0.007000000216)/0.08341437578
		outputs[149] = (inputs[149]-0.007000000216)/0.08341437578
		outputs[150] = (inputs[150]-0.007000000216)/0.08341437578
		outputs[151] = (inputs[151]-0.007000000216)/0.08341437578
		outputs[152] = (inputs[152]-0.007000000216)/0.08341437578
		outputs[153] = (inputs[153]-0.007000000216)/0.08341437578
		outputs[154] = (inputs[154]-0.007000000216)/0.08341437578
		outputs[155] = (inputs[155]-0.007000000216)/0.08341437578
		outputs[156] = (inputs[156]-0.007000000216)/0.08341437578
		outputs[157] = (inputs[157]-0.007000000216)/0.08341437578
		outputs[158] = (inputs[158]-0.007000000216)/0.08341437578
		outputs[159] = (inputs[159]-0.007000000216)/0.08341437578
		outputs[160] = (inputs[160]-0.007000000216)/0.08341437578
		outputs[161] = (inputs[161]-0.007000000216)/0.08341437578
		outputs[162] = (inputs[162]-0.007000000216)/0.08341437578
		outputs[163] = (inputs[163]-0.007000000216)/0.08341437578
		outputs[164] = (inputs[164]-0.007000000216)/0.08341437578
		outputs[165] = (inputs[165]-0.007000000216)/0.09465706348
		outputs[166] = (inputs[166]-0.007000000216)/0.08341437578
		outputs[167] = (inputs[167]-0.007000000216)/0.08341437578
		outputs[168] = (inputs[168]-0.007000000216)/0.08341437578
		outputs[169] = (inputs[169]-0.007000000216)/0.08341437578
		outputs[170] = (inputs[170]-0.007000000216)/0.08341437578
		outputs[171] = (inputs[171]-0.007000000216)/0.08341437578
		outputs[172] = (inputs[172]-0.007000000216)/0.08341437578
		outputs[173] = (inputs[173]-0.007000000216)/0.08341437578
		outputs[174] = (inputs[174]-0.007000000216)/0.08341437578
		outputs[175] = (inputs[175]-0.007000000216)/0.08341437578
		outputs[176] = (inputs[176]-0.006000000052)/0.07726558298
		outputs[177] = (inputs[177]-0.006000000052)/0.07726558298
		outputs[178] = (inputs[178]-0.006000000052)/0.07726558298
		outputs[179] = (inputs[179]-0.006000000052)/0.07726558298
		outputs[180] = (inputs[180]-0.006000000052)/0.07726558298
		outputs[181] = (inputs[181]-0.006000000052)/0.07726558298
		outputs[182] = (inputs[182]-0.006000000052)/0.08928590268
		outputs[183] = (inputs[183]-0.006000000052)/0.07726558298
		outputs[184] = (inputs[184]-0.006000000052)/0.07726558298
		outputs[185] = (inputs[185]-0.006000000052)/0.07726558298
		outputs[186] = (inputs[186]-0.006000000052)/0.07726558298
		outputs[187] = (inputs[187]-0.006000000052)/0.07726558298
		outputs[188] = (inputs[188]-0.006000000052)/0.07726558298
		outputs[189] = (inputs[189]-0.006000000052)/0.07726558298
		outputs[190] = (inputs[190]-0.006000000052)/0.07726558298
		outputs[191] = (inputs[191]-0.006000000052)/0.07726558298
		outputs[192] = (inputs[192]-0.006000000052)/0.08928590268
		outputs[193] = (inputs[193]-0.006000000052)/0.07726558298
		outputs[194] = (inputs[194]-0.006000000052)/0.07726558298
		outputs[195] = (inputs[195]-0.006000000052)/0.07726558298
		outputs[196] = (inputs[196]-0.006000000052)/0.07726558298
		outputs[197] = (inputs[197]-0.006000000052)/0.07726558298
		outputs[198] = (inputs[198]-0.006000000052)/0.07726558298
		outputs[199] = (inputs[199]-0.006000000052)/0.07726558298
		outputs[200] = (inputs[200]-0.006000000052)/0.07726558298
		outputs[201] = (inputs[201]-0.006000000052)/0.07726558298
		outputs[202] = (inputs[202]-0.006000000052)/0.07726558298
		outputs[203] = (inputs[203]-0.006000000052)/0.07726558298
		outputs[204] = (inputs[204]-0.006000000052)/0.07726558298
		outputs[205] = (inputs[205]-0.004999999888)/0.07056897134
		outputs[206] = (inputs[206]-0.004999999888)/0.07056897134
		outputs[207] = (inputs[207]-0.004999999888)/0.07056897134
		outputs[208] = (inputs[208]-0.004999999888)/0.07056897134
		outputs[209] = (inputs[209]-0.004999999888)/0.07056897134
		outputs[210] = (inputs[210]-0.004999999888)/0.07056897134
		outputs[211] = (inputs[211]-0.004999999888)/0.07056897134
		outputs[212] = (inputs[212]-0.004999999888)/0.07056897134
		outputs[213] = (inputs[213]-0.004999999888)/0.07056897134
		outputs[214] = (inputs[214]-0.004999999888)/0.07056897134
		outputs[215] = (inputs[215]-0.004999999888)/0.08355825394
		outputs[216] = (inputs[216]-0.004999999888)/0.07056897134
		outputs[217] = (inputs[217]-0.004999999888)/0.07056897134
		outputs[218] = (inputs[218]-0.004999999888)/0.07056897134
		outputs[219] = (inputs[219]-0.004999999888)/0.07056897134
		outputs[220] = (inputs[220]-0.004999999888)/0.07056897134
		outputs[221] = (inputs[221]-0.004999999888)/0.07056897134
		outputs[222] = (inputs[222]-0.004999999888)/0.07056897134
		outputs[223] = (inputs[223]-0.004999999888)/0.07056897134
		outputs[224] = (inputs[224]-0.004999999888)/0.07056897134
		outputs[225] = (inputs[225]-0.004999999888)/0.07056897134
		outputs[226] = (inputs[226]-0.004999999888)/0.07056897134
		outputs[227] = (inputs[227]-0.004999999888)/0.07056897134
		outputs[228] = (inputs[228]-0.004999999888)/0.07056897134
		outputs[229] = (inputs[229]-0.004999999888)/0.07056897134
		outputs[230] = (inputs[230]-0.004999999888)/0.07056897134
		outputs[231] = (inputs[231]-0.004999999888)/0.07056897134
		outputs[232] = (inputs[232]-0.004999999888)/0.08355825394
		outputs[233] = (inputs[233]-0.004999999888)/0.07056897134
		outputs[234] = (inputs[234]-0.004999999888)/0.07056897134
		outputs[235] = (inputs[235]-0.004999999888)/0.07056897134
		outputs[236] = (inputs[236]-0.004999999888)/0.07056897134
		outputs[237] = (inputs[237]-0.004999999888)/0.07056897134
		outputs[238] = (inputs[238]-0.004999999888)/0.07056897134
		outputs[239] = (inputs[239]-0.004999999888)/0.07056897134
		outputs[240] = (inputs[240]-0.004999999888)/0.07056897134
		outputs[241] = (inputs[241]-0.004999999888)/0.07056897134
		outputs[242] = (inputs[242]-0.004999999888)/0.07056897134
		outputs[243] = (inputs[243]-0.004999999888)/0.07056897134
		outputs[244] = (inputs[244]-0.004999999888)/0.07056897134
		outputs[245] = (inputs[245]-0.004999999888)/0.07056897134
		outputs[246] = (inputs[246]-0.004999999888)/0.07056897134
		outputs[247] = (inputs[247]-0.004999999888)/0.07056897134
		outputs[248] = (inputs[248]-0.004999999888)/0.07056897134
		outputs[249] = (inputs[249]-0.004999999888)/0.07056897134
		outputs[250] = (inputs[250]-0.004999999888)/0.08355825394
		outputs[251] = (inputs[251]-0.004999999888)/0.07056897134
		outputs[252] = (inputs[252]-0.004999999888)/0.07056897134
		outputs[253] = (inputs[253]-0.004999999888)/0.07056897134
		outputs[254] = (inputs[254]-0.00400000019)/0.06315051764
		outputs[255] = (inputs[255]-0.00400000019)/0.06315051764
		outputs[256] = (inputs[256]-0.00400000019)/0.06315051764
		outputs[257] = (inputs[257]-0.00400000019)/0.06315051764
		outputs[258] = (inputs[258]-0.00400000019)/0.06315051764
		outputs[259] = (inputs[259]-0.00400000019)/0.06315051764
		outputs[260] = (inputs[260]-0.00400000019)/0.06315051764
		outputs[261] = (inputs[261]-0.00400000019)/0.06315051764
		outputs[262] = (inputs[262]-0.00400000019)/0.06315051764
		outputs[263] = (inputs[263]-0.00400000019)/0.06315051764
		outputs[264] = (inputs[264]-0.00400000019)/0.06315051764
		outputs[265] = (inputs[265]-0.00400000019)/0.06315051764
		outputs[266] = (inputs[266]-0.00400000019)/0.06315051764
		outputs[267] = (inputs[267]-0.00400000019)/0.06315051764
		outputs[268] = (inputs[268]-0.00400000019)/0.06315051764
		outputs[269] = (inputs[269]-0.00400000019)/0.06315051764
		outputs[270] = (inputs[270]-0.00400000019)/0.06315051764
		outputs[271] = (inputs[271]-0.00400000019)/0.06315051764
		outputs[272] = (inputs[272]-0.00400000019)/0.06315051764
		outputs[273] = (inputs[273]-0.00400000019)/0.06315051764
		outputs[274] = (inputs[274]-0.00400000019)/0.06315051764
		outputs[275] = (inputs[275]-0.00400000019)/0.06315051764
		outputs[276] = (inputs[276]-0.00400000019)/0.06315051764
		outputs[277] = (inputs[277]-0.00400000019)/0.06315051764
		outputs[278] = (inputs[278]-0.00400000019)/0.06315051764
		outputs[279] = (inputs[279]-0.00400000019)/0.06315051764
		outputs[280] = (inputs[280]-0.00400000019)/0.06315051764
		outputs[281] = (inputs[281]-0.00400000019)/0.06315051764
		outputs[282] = (inputs[282]-0.00400000019)/0.06315051764
		outputs[283] = (inputs[283]-0.00400000019)/0.06315051764
		outputs[284] = (inputs[284]-0.00400000019)/0.06315051764
		outputs[285] = (inputs[285]-0.00400000019)/0.06315051764
		outputs[286] = (inputs[286]-0.00400000019)/0.06315051764
		outputs[287] = (inputs[287]-0.00400000019)/0.06315051764
		outputs[288] = (inputs[288]-0.00400000019)/0.06315051764
		outputs[289] = (inputs[289]-0.00400000019)/0.06315051764
		outputs[290] = (inputs[290]-0.00400000019)/0.06315051764
		outputs[291] = (inputs[291]-0.00400000019)/0.06315051764
		outputs[292] = (inputs[292]-0.00400000019)/0.06315051764
		outputs[293] = (inputs[293]-0.00400000019)/0.06315051764
		outputs[294] = (inputs[294]-0.00400000019)/0.0999699682
		outputs[295] = (inputs[295]-0.00400000019)/0.06315051764
		outputs[296] = (inputs[296]-0.00400000019)/0.06315051764
		outputs[297] = (inputs[297]-0.00400000019)/0.06315051764
		outputs[298] = (inputs[298]-0.00400000019)/0.06315051764
		outputs[299] = (inputs[299]-0.00400000019)/0.06315051764
		outputs[300] = (inputs[300]-0.00400000019)/0.06315051764
		outputs[301] = (inputs[301]-0.00400000019)/0.06315051764
		outputs[302] = (inputs[302]-0.00400000019)/0.06315051764
		outputs[303] = (inputs[303]-0.00400000019)/0.06315051764
		outputs[304] = (inputs[304]-0.00400000019)/0.06315051764
		outputs[305] = (inputs[305]-0.00400000019)/0.06315051764
		outputs[306] = (inputs[306]-0.00400000019)/0.06315051764
		outputs[307] = (inputs[307]-0.00400000019)/0.06315051764
		outputs[308] = (inputs[308]-0.00400000019)/0.06315051764
		outputs[309] = (inputs[309]-0.00400000019)/0.06315051764
		outputs[310] = (inputs[310]-0.00400000019)/0.06315051764
		outputs[311] = (inputs[311]-0.00400000019)/0.06315051764
		outputs[312] = (inputs[312]-0.00400000019)/0.06315051764
		outputs[313] = (inputs[313]-0.00400000019)/0.06315051764
		outputs[314] = (inputs[314]-0.00400000019)/0.07739502192
		outputs[315] = (inputs[315]-0.00400000019)/0.06315051764
		outputs[316] = (inputs[316]-0.00400000019)/0.06315051764
		outputs[317] = (inputs[317]-0.00400000019)/0.06315051764
		outputs[318] = (inputs[318]-0.003000000026)/0.05471740291
		outputs[319] = (inputs[319]-0.003000000026)/0.05471740291
		outputs[320] = (inputs[320]-0.003000000026)/0.05471740291
		outputs[321] = (inputs[321]-0.003000000026)/0.05471740291
		outputs[322] = (inputs[322]-0.003000000026)/0.07068236172
		outputs[323] = (inputs[323]-0.003000000026)/0.05471740291
		outputs[324] = (inputs[324]-0.003000000026)/0.05471740291
		outputs[325] = (inputs[325]-0.003000000026)/0.07068236172
		outputs[326] = (inputs[326]-0.003000000026)/0.05471740291
		outputs[327] = (inputs[327]-0.003000000026)/0.07068236172
		outputs[328] = (inputs[328]-0.003000000026)/0.05471740291
		outputs[329] = (inputs[329]-0.003000000026)/0.05471740291
		outputs[330] = (inputs[330]-0.003000000026)/0.05471740291
		outputs[331] = (inputs[331]-0.003000000026)/0.05471740291
		outputs[332] = (inputs[332]-0.003000000026)/0.05471740291
		outputs[333] = (inputs[333]-0.003000000026)/0.05471740291
		outputs[334] = (inputs[334]-0.003000000026)/0.05471740291
		outputs[335] = (inputs[335]-0.003000000026)/0.05471740291
		outputs[336] = (inputs[336]-0.003000000026)/0.05471740291
		outputs[337] = (inputs[337]-0.003000000026)/0.05471740291
		outputs[338] = (inputs[338]-0.003000000026)/0.05471740291
		outputs[339] = (inputs[339]-0.003000000026)/0.05471740291
		outputs[340] = (inputs[340]-0.003000000026)/0.07068236172
		outputs[341] = (inputs[341]-0.003000000026)/0.05471740291
		outputs[342] = (inputs[342]-0.003000000026)/0.05471740291
		outputs[343] = (inputs[343]-0.003000000026)/0.05471740291
		outputs[344] = (inputs[344]-0.003000000026)/0.05471740291
		outputs[345] = (inputs[345]-0.003000000026)/0.05471740291
		outputs[346] = (inputs[346]-0.003000000026)/0.05471740291
		outputs[347] = (inputs[347]-0.003000000026)/0.05471740291
		outputs[348] = (inputs[348]-0.003000000026)/0.05471740291
		outputs[349] = (inputs[349]-0.003000000026)/0.05471740291
		outputs[350] = (inputs[350]-0.003000000026)/0.05471740291
		outputs[351] = (inputs[351]-0.003000000026)/0.05471740291
		outputs[352] = (inputs[352]-0.003000000026)/0.05471740291
		outputs[353] = (inputs[353]-0.003000000026)/0.05471740291
		outputs[354] = (inputs[354]-0.003000000026)/0.05471740291
		outputs[355] = (inputs[355]-0.003000000026)/0.05471740291
		outputs[356] = (inputs[356]-0.003000000026)/0.05471740291
		outputs[357] = (inputs[357]-0.003000000026)/0.05471740291
		outputs[358] = (inputs[358]-0.003000000026)/0.05471740291
		outputs[359] = (inputs[359]-0.003000000026)/0.05471740291
		outputs[360] = (inputs[360]-0.003000000026)/0.05471740291
		outputs[361] = (inputs[361]-0.003000000026)/0.05471740291
		outputs[362] = (inputs[362]-0.003000000026)/0.05471740291
		outputs[363] = (inputs[363]-0.003000000026)/0.05471740291
		outputs[364] = (inputs[364]-0.003000000026)/0.05471740291
		outputs[365] = (inputs[365]-0.003000000026)/0.05471740291
		outputs[366] = (inputs[366]-0.003000000026)/0.05471740291
		outputs[367] = (inputs[367]-0.003000000026)/0.05471740291
		outputs[368] = (inputs[368]-0.003000000026)/0.05471740291
		outputs[369] = (inputs[369]-0.003000000026)/0.05471740291
		outputs[370] = (inputs[370]-0.003000000026)/0.05471740291
		outputs[371] = (inputs[371]-0.003000000026)/0.05471740291
		outputs[372] = (inputs[372]-0.003000000026)/0.05471740291
		outputs[373] = (inputs[373]-0.003000000026)/0.05471740291
		outputs[374] = (inputs[374]-0.003000000026)/0.05471740291
		outputs[375] = (inputs[375]-0.003000000026)/0.05471740291
		outputs[376] = (inputs[376]-0.003000000026)/0.05471740291
		outputs[377] = (inputs[377]-0.003000000026)/0.05471740291
		outputs[378] = (inputs[378]-0.003000000026)/0.05471740291
		outputs[379] = (inputs[379]-0.003000000026)/0.05471740291
		outputs[380] = (inputs[380]-0.003000000026)/0.05471740291
		outputs[381] = (inputs[381]-0.003000000026)/0.05471740291
		outputs[382] = (inputs[382]-0.003000000026)/0.05471740291
		outputs[383] = (inputs[383]-0.003000000026)/0.05471740291
		outputs[384] = (inputs[384]-0.003000000026)/0.05471740291
		outputs[385] = (inputs[385]-0.003000000026)/0.05471740291
		outputs[386] = (inputs[386]-0.003000000026)/0.05471740291
		outputs[387] = (inputs[387]-0.003000000026)/0.05471740291
		outputs[388] = (inputs[388]-0.003000000026)/0.05471740291
		outputs[389] = (inputs[389]-0.003000000026)/0.05471740291
		outputs[390] = (inputs[390]-0.003000000026)/0.05471740291
		outputs[391] = (inputs[391]-0.003000000026)/0.05471740291
		outputs[392] = (inputs[392]-0.003000000026)/0.05471740291
		outputs[393] = (inputs[393]-0.003000000026)/0.05471740291
		outputs[394] = (inputs[394]-0.003000000026)/0.05471740291
		outputs[395] = (inputs[395]-0.003000000026)/0.05471740291
		outputs[396] = (inputs[396]-0.003000000026)/0.05471740291
		outputs[397] = (inputs[397]-0.003000000026)/0.05471740291
		outputs[398] = (inputs[398]-0.003000000026)/0.05471740291
		outputs[399] = (inputs[399]-0.003000000026)/0.05471740291
		outputs[400] = (inputs[400]-0.003000000026)/0.05471740291
		outputs[401] = (inputs[401]-0.003000000026)/0.05471740291
		outputs[402] = (inputs[402]-0.003000000026)/0.05471740291
		outputs[403] = (inputs[403]-0.003000000026)/0.05471740291
		outputs[404] = (inputs[404]-0.003000000026)/0.05471740291
		outputs[405] = (inputs[405]-0.003000000026)/0.05471740291
		outputs[406] = (inputs[406]-0.003000000026)/0.05471740291
		outputs[407] = (inputs[407]-0.003000000026)/0.05471740291
		outputs[408] = (inputs[408]-0.003000000026)/0.05471740291
		outputs[409] = (inputs[409]-0.003000000026)/0.05471740291
		outputs[410] = (inputs[410]-0.003000000026)/0.05471740291
		outputs[411] = (inputs[411]-0.003000000026)/0.05471740291
		outputs[412] = (inputs[412]-0.003000000026)/0.05471740291
		outputs[413] = (inputs[413]-0.003000000026)/0.05471740291
		outputs[414] = (inputs[414]-0.003000000026)/0.05471740291
		outputs[415] = (inputs[415]-0.003000000026)/0.05471740291
		outputs[416] = (inputs[416]-0.003000000026)/0.07068236172
		outputs[417] = (inputs[417]-0.003000000026)/0.05471740291
		outputs[418] = (inputs[418]-0.003000000026)/0.05471740291
		outputs[419] = (inputs[419]-0.003000000026)/0.05471740291
		outputs[420] = (inputs[420]-0.003000000026)/0.05471740291
		outputs[421] = (inputs[421]-0.003000000026)/0.05471740291
		outputs[422] = (inputs[422]-0.003000000026)/0.05471740291
		outputs[423] = (inputs[423]-0.003000000026)/0.05471740291
		outputs[424] = (inputs[424]-0.003000000026)/0.05471740291
		outputs[425] = (inputs[425]-0.003000000026)/0.05471740291
		outputs[426] = (inputs[426]-0.003000000026)/0.05471740291
		outputs[427] = (inputs[427]-0.003000000026)/0.05471740291
		outputs[428] = (inputs[428]-0.003000000026)/0.05471740291
		outputs[429] = (inputs[429]-0.003000000026)/0.05471740291
		outputs[430] = (inputs[430]-0.003000000026)/0.05471740291
		outputs[431] = (inputs[431]-0.003000000026)/0.05471740291
		outputs[432] = (inputs[432]-0.003000000026)/0.05471740291
		outputs[433] = (inputs[433]-0.003000000026)/0.05471740291
		outputs[434] = (inputs[434]-0.003000000026)/0.05471740291
		outputs[435] = (inputs[435]-0.003000000026)/0.05471740291
		outputs[436] = (inputs[436]-0.003000000026)/0.05471740291
		outputs[437] = (inputs[437]-0.003000000026)/0.05471740291
		outputs[438] = (inputs[438]-0.003000000026)/0.05471740291
		outputs[439] = (inputs[439]-0.003000000026)/0.05471740291
		outputs[440] = (inputs[440]-0.003000000026)/0.05471740291
		outputs[441] = (inputs[441]-0.003000000026)/0.05471740291
		outputs[442] = (inputs[442]-0.003000000026)/0.05471740291
		outputs[443] = (inputs[443]-0.003000000026)/0.05471740291
		outputs[444] = (inputs[444]-0.003000000026)/0.05471740291
		outputs[445] = (inputs[445]-0.003000000026)/0.05471740291
		outputs[446] = (inputs[446]-0.003000000026)/0.05471740291
		outputs[447] = (inputs[447]-0.003000000026)/0.05471740291
		outputs[448] = (inputs[448]-0.003000000026)/0.05471740291
		outputs[449] = (inputs[449]-0.002000000095)/0.04469897225
		outputs[450] = (inputs[450]-0.002000000095)/0.04469897225
		outputs[451] = (inputs[451]-0.002000000095)/0.04469897225
		outputs[452] = (inputs[452]-0.002000000095)/0.04469897225
		outputs[453] = (inputs[453]-0.002000000095)/0.04469897225
		outputs[454] = (inputs[454]-0.002000000095)/0.04469897225
		outputs[455] = (inputs[455]-0.002000000095)/0.04469897225
		outputs[456] = (inputs[456]-0.002000000095)/0.04469897225
		outputs[457] = (inputs[457]-0.002000000095)/0.04469897225
		outputs[458] = (inputs[458]-0.002000000095)/0.04469897225
		outputs[459] = (inputs[459]-0.002000000095)/0.04469897225
		outputs[460] = (inputs[460]-0.002000000095)/0.04469897225
		outputs[461] = (inputs[461]-0.002000000095)/0.04469897225
		outputs[462] = (inputs[462]-0.002000000095)/0.04469897225
		outputs[463] = (inputs[463]-0.002000000095)/0.04469897225
		outputs[464] = (inputs[464]-0.002000000095)/0.04469897225
		outputs[465] = (inputs[465]-0.002000000095)/0.04469897225
		outputs[466] = (inputs[466]-0.002000000095)/0.04469897225
		outputs[467] = (inputs[467]-0.002000000095)/0.04469897225
		outputs[468] = (inputs[468]-0.002000000095)/0.04469897225
		outputs[469] = (inputs[469]-0.002000000095)/0.04469897225
		outputs[470] = (inputs[470]-0.002000000095)/0.04469897225
		outputs[471] = (inputs[471]-0.002000000095)/0.04469897225
		outputs[472] = (inputs[472]-0.002000000095)/0.04469897225
		outputs[473] = (inputs[473]-0.002000000095)/0.04469897225
		outputs[474] = (inputs[474]-0.002000000095)/0.04469897225
		outputs[475] = (inputs[475]-0.002000000095)/0.04469897225
		outputs[476] = (inputs[476]-0.001000000047)/0.0316227749
		outputs[477] = (inputs[477]-0.002000000095)/0.04469897225
		outputs[478] = (inputs[478]-0.002000000095)/0.04469897225
		outputs[479] = (inputs[479]-0.002000000095)/0.04469897225
		outputs[480] = (inputs[480]-0.002000000095)/0.04469897225
		outputs[481] = (inputs[481]-0.002000000095)/0.04469897225
		outputs[482] = (inputs[482]-0.002000000095)/0.04469897225
		outputs[483] = (inputs[483]-0.002000000095)/0.04469897225
		outputs[484] = (inputs[484]-0.002000000095)/0.04469897225
		outputs[485] = (inputs[485]-0.002000000095)/0.04469897225
		outputs[486] = (inputs[486]-0.002000000095)/0.04469897225
		outputs[487] = (inputs[487]-0.002000000095)/0.04469897225
		outputs[488] = (inputs[488]-0.002000000095)/0.04469897225
		outputs[489] = (inputs[489]-0.002000000095)/0.04469897225
		outputs[490] = (inputs[490]-0.002000000095)/0.04469897225
		outputs[491] = (inputs[491]-0.002000000095)/0.04469897225
		outputs[492] = (inputs[492]-0.002000000095)/0.04469897225
		outputs[493] = (inputs[493]-0.002000000095)/0.04469897225
		outputs[494] = (inputs[494]-0.002000000095)/0.04469897225
		outputs[495] = (inputs[495]-0.002000000095)/0.04469897225
		outputs[496] = (inputs[496]-0.002000000095)/0.04469897225
		outputs[497] = (inputs[497]-0.002000000095)/0.04469897225
		outputs[498] = (inputs[498]-0.002000000095)/0.04469897225
		outputs[499] = (inputs[499]-0.002000000095)/0.04469897225
		outputs[500] = (inputs[500]-0.002000000095)/0.04469897225
		outputs[501] = (inputs[501]-0.002000000095)/0.04469897225
		outputs[502] = (inputs[502]-0.002000000095)/0.04469897225
		outputs[503] = (inputs[503]-0.002000000095)/0.04469897225
		outputs[504] = (inputs[504]-0.002000000095)/0.04469897225
		outputs[505] = (inputs[505]-0.002000000095)/0.04469897225
		outputs[506] = (inputs[506]-0.002000000095)/0.04469897225
		outputs[507] = (inputs[507]-0.002000000095)/0.04469897225
		outputs[508] = (inputs[508]-0.002000000095)/0.04469897225
		outputs[509] = (inputs[509]-0.002000000095)/0.04469897225
		outputs[510] = (inputs[510]-0.002000000095)/0.04469897225
		outputs[511] = (inputs[511]-0.002000000095)/0.04469897225
		outputs[512] = (inputs[512]-0.002000000095)/0.04469897225
		outputs[513] = (inputs[513]-0.002000000095)/0.04469897225
		outputs[514] = (inputs[514]-0.002000000095)/0.04469897225
		outputs[515] = (inputs[515]-0.002000000095)/0.04469897225
		outputs[516] = (inputs[516]-0.002000000095)/0.04469897225
		outputs[517] = (inputs[517]-0.002000000095)/0.04469897225
		outputs[518] = (inputs[518]-0.002000000095)/0.04469897225
		outputs[519] = (inputs[519]-0.002000000095)/0.04469897225
		outputs[520] = (inputs[520]-0.002000000095)/0.04469897225
		outputs[521] = (inputs[521]-0.002000000095)/0.04469897225
		outputs[522] = (inputs[522]-0.002000000095)/0.04469897225
		outputs[523] = (inputs[523]-0.002000000095)/0.04469897225
		outputs[524] = (inputs[524]-0.002000000095)/0.04469897225
		outputs[525] = (inputs[525]-0.002000000095)/0.04469897225
		outputs[526] = (inputs[526]-0.002000000095)/0.04469897225
		outputs[527] = (inputs[527]-0.002000000095)/0.04469897225
		outputs[528] = (inputs[528]-0.002000000095)/0.04469897225
		outputs[529] = (inputs[529]-0.002000000095)/0.04469897225
		outputs[530] = (inputs[530]-0.002000000095)/0.04469897225
		outputs[531] = (inputs[531]-0.002000000095)/0.04469897225
		outputs[532] = (inputs[532]-0.002000000095)/0.04469897225
		outputs[533] = (inputs[533]-0.002000000095)/0.04469897225
		outputs[534] = (inputs[534]-0.002000000095)/0.04469897225
		outputs[535] = (inputs[535]-0.001000000047)/0.0316227749
		outputs[536] = (inputs[536]-0.002000000095)/0.04469897225
		outputs[537] = (inputs[537]-0.002000000095)/0.04469897225
		outputs[538] = (inputs[538]-0.002000000095)/0.04469897225
		outputs[539] = (inputs[539]-0.001000000047)/0.0316227749
		outputs[540] = (inputs[540]-0.002000000095)/0.04469897225
		outputs[541] = (inputs[541]-0.002000000095)/0.04469897225
		outputs[542] = (inputs[542]-0.002000000095)/0.04469897225
		outputs[543] = (inputs[543]-0.002000000095)/0.04469897225
		outputs[544] = (inputs[544]-0.002000000095)/0.04469897225
		outputs[545] = (inputs[545]-0.002000000095)/0.04469897225
		outputs[546] = (inputs[546]-0.002000000095)/0.04469897225
		outputs[547] = (inputs[547]-0.002000000095)/0.04469897225
		outputs[548] = (inputs[548]-0.002000000095)/0.04469897225
		outputs[549] = (inputs[549]-0.002000000095)/0.04469897225
		outputs[550] = (inputs[550]-0.002000000095)/0.04469897225
		outputs[551] = (inputs[551]-0.002000000095)/0.04469897225
		outputs[552] = (inputs[552]-0.002000000095)/0.04469897225
		outputs[553] = (inputs[553]-0.002000000095)/0.04469897225
		outputs[554] = (inputs[554]-0.002000000095)/0.04469897225
		outputs[555] = (inputs[555]-0.002000000095)/0.04469897225
		outputs[556] = (inputs[556]-0.002000000095)/0.04469897225
		outputs[557] = (inputs[557]-0.002000000095)/0.04469897225
		outputs[558] = (inputs[558]-0.002000000095)/0.04469897225
		outputs[559] = (inputs[559]-0.002000000095)/0.04469897225
		outputs[560] = (inputs[560]-0.002000000095)/0.04469897225
		outputs[561] = (inputs[561]-0.002000000095)/0.04469897225
		outputs[562] = (inputs[562]-0.002000000095)/0.04469897225
		outputs[563] = (inputs[563]-0.002000000095)/0.04469897225
		outputs[564] = (inputs[564]-0.002000000095)/0.04469897225
		outputs[565] = (inputs[565]-0.002000000095)/0.04469897225
		outputs[566] = (inputs[566]-0.002000000095)/0.04469897225
		outputs[567] = (inputs[567]-0.002000000095)/0.04469897225
		outputs[568] = (inputs[568]-0.002000000095)/0.04469897225
		outputs[569] = (inputs[569]-0.002000000095)/0.04469897225
		outputs[570] = (inputs[570]-0.002000000095)/0.04469897225
		outputs[571] = (inputs[571]-0.002000000095)/0.04469897225
		outputs[572] = (inputs[572]-0.002000000095)/0.04469897225
		outputs[573] = (inputs[573]-0.002000000095)/0.04469897225
		outputs[574] = (inputs[574]-0.002000000095)/0.04469897225
		outputs[575] = (inputs[575]-0.002000000095)/0.04469897225
		outputs[576] = (inputs[576]-0.002000000095)/0.04469897225
		outputs[577] = (inputs[577]-0.002000000095)/0.04469897225
		outputs[578] = (inputs[578]-0.002000000095)/0.04469897225
		outputs[579] = (inputs[579]-0.002000000095)/0.04469897225
		outputs[580] = (inputs[580]-0.002000000095)/0.04469897225
		outputs[581] = (inputs[581]-0.002000000095)/0.04469897225
		outputs[582] = (inputs[582]-0.002000000095)/0.04469897225
		outputs[583] = (inputs[583]-0.002000000095)/0.04469897225
		outputs[584] = (inputs[584]-0.002000000095)/0.04469897225
		outputs[585] = (inputs[585]-0.002000000095)/0.04469897225
		outputs[586] = (inputs[586]-0.002000000095)/0.04469897225
		outputs[587] = (inputs[587]-0.002000000095)/0.04469897225
		outputs[588] = (inputs[588]-0.002000000095)/0.04469897225
		outputs[589] = (inputs[589]-0.002000000095)/0.04469897225
		outputs[590] = (inputs[590]-0.002000000095)/0.04469897225
		outputs[591] = (inputs[591]-0.001000000047)/0.0316227749
		outputs[592] = (inputs[592]-0.002000000095)/0.04469897225
		outputs[593] = (inputs[593]-0.002000000095)/0.04469897225
		outputs[594] = (inputs[594]-0.002000000095)/0.04469897225
		outputs[595] = (inputs[595]-0.002000000095)/0.04469897225
		outputs[596] = (inputs[596]-0.002000000095)/0.04469897225
		outputs[597] = (inputs[597]-0.002000000095)/0.04469897225
		outputs[598] = (inputs[598]-0.002000000095)/0.04469897225
		outputs[599] = (inputs[599]-0.001000000047)/0.0316227749
		outputs[600] = (inputs[600]-0.002000000095)/0.04469897225
		outputs[601] = (inputs[601]-0.002000000095)/0.04469897225
		outputs[602] = (inputs[602]-0.002000000095)/0.04469897225
		outputs[603] = (inputs[603]-0.002000000095)/0.04469897225
		outputs[604] = (inputs[604]-0.002000000095)/0.04469897225
		outputs[605] = (inputs[605]-0.002000000095)/0.04469897225
		outputs[606] = (inputs[606]-0.002000000095)/0.04469897225
		outputs[607] = (inputs[607]-0.002000000095)/0.04469897225
		outputs[608] = (inputs[608]-0.002000000095)/0.04469897225
		outputs[609] = (inputs[609]-0.002000000095)/0.04469897225
		outputs[610] = (inputs[610]-0.002000000095)/0.04469897225
		outputs[611] = (inputs[611]-0.002000000095)/0.04469897225
		outputs[612] = (inputs[612]-0.002000000095)/0.04469897225
		outputs[613] = (inputs[613]-0.002000000095)/0.04469897225
		outputs[614] = (inputs[614]-0.001000000047)/0.0316227749
		outputs[615] = (inputs[615]-0.002000000095)/0.04469897225
		outputs[616] = (inputs[616]-0.002000000095)/0.04469897225
		outputs[617] = (inputs[617]-0.002000000095)/0.04469897225
		outputs[618] = (inputs[618]-0.002000000095)/0.04469897225
		outputs[619] = (inputs[619]-0.002000000095)/0.04469897225
		outputs[620] = (inputs[620]-0.002000000095)/0.04469897225
		outputs[621] = (inputs[621]-0.002000000095)/0.04469897225
		outputs[622] = (inputs[622]-0.002000000095)/0.04469897225
		outputs[623] = (inputs[623]-0.002000000095)/0.04469897225
		outputs[624] = (inputs[624]-0.002000000095)/0.04469897225
		outputs[625] = (inputs[625]-0.002000000095)/0.04469897225
		outputs[626] = (inputs[626]-0.002000000095)/0.04469897225
		outputs[627] = (inputs[627]-0.002000000095)/0.04469897225
		outputs[628] = (inputs[628]-0.002000000095)/0.04469897225
		outputs[629] = (inputs[629]-0.002000000095)/0.04469897225
		outputs[630] = (inputs[630]-0.002000000095)/0.04469897225
		outputs[631] = (inputs[631]-0.002000000095)/0.04469897225
		outputs[632] = (inputs[632]-0.002000000095)/0.04469897225
		outputs[633] = (inputs[633]-0.002000000095)/0.04469897225
		outputs[634] = (inputs[634]-0.002000000095)/0.04469897225
		outputs[635] = (inputs[635]-0.002000000095)/0.04469897225
		outputs[636] = (inputs[636]-0.002000000095)/0.04469897225
		outputs[637] = (inputs[637]-0.002000000095)/0.04469897225
		outputs[638] = (inputs[638]-0.002000000095)/0.04469897225
		outputs[639] = (inputs[639]-0.002000000095)/0.04469897225
		outputs[640] = (inputs[640]-0.002000000095)/0.04469897225
		outputs[641] = (inputs[641]-0.002000000095)/0.04469897225
		outputs[642] = (inputs[642]-0.002000000095)/0.04469897225
		outputs[643] = (inputs[643]-0.002000000095)/0.04469897225
		outputs[644] = (inputs[644]-0.002000000095)/0.04469897225
		outputs[645] = (inputs[645]-0.002000000095)/0.04469897225
		outputs[646] = (inputs[646]-0.002000000095)/0.04469897225
		outputs[647] = (inputs[647]-0.002000000095)/0.04469897225
		outputs[648] = (inputs[648]-0.001000000047)/0.0316227749
		outputs[649] = (inputs[649]-0.001000000047)/0.0316227749
		outputs[650] = (inputs[650]-0.001000000047)/0.0316227749
		outputs[651] = (inputs[651]-0.001000000047)/0.0316227749
		outputs[652] = (inputs[652]-0.001000000047)/0.0316227749
		outputs[653] = (inputs[653]-0.001000000047)/0.0316227749
		outputs[654] = (inputs[654]-0.001000000047)/0.0316227749
		outputs[655] = (inputs[655]-0.001000000047)/0.0316227749
		outputs[656] = (inputs[656]-0.001000000047)/0.0316227749
		outputs[657] = (inputs[657]-0.001000000047)/0.0316227749
		outputs[658] = (inputs[658]-0.001000000047)/0.0316227749
		outputs[659] = (inputs[659]-0.001000000047)/0.0316227749
		outputs[660] = (inputs[660]-0.001000000047)/0.0316227749
		outputs[661] = (inputs[661]-0.001000000047)/0.0316227749
		outputs[662] = (inputs[662]-0.001000000047)/0.0316227749
		outputs[663] = (inputs[663]-0.001000000047)/0.0316227749
		outputs[664] = (inputs[664]-0.001000000047)/0.0316227749
		outputs[665] = (inputs[665]-0.001000000047)/0.0316227749
		outputs[666] = (inputs[666]-0.001000000047)/0.0316227749
		outputs[667] = (inputs[667]-0.001000000047)/0.0316227749
		outputs[668] = (inputs[668]-0.001000000047)/0.0316227749
		outputs[669] = (inputs[669]-0.001000000047)/0.0316227749
		outputs[670] = (inputs[670]-0.001000000047)/0.0316227749
		outputs[671] = (inputs[671]-0.001000000047)/0.0316227749
		outputs[672] = (inputs[672]-0.001000000047)/0.0316227749
		outputs[673] = (inputs[673]-0.001000000047)/0.0316227749
		outputs[674] = (inputs[674]-0.001000000047)/0.0316227749
		outputs[675] = (inputs[675]-0.001000000047)/0.0316227749
		outputs[676] = (inputs[676]-0.001000000047)/0.0316227749
		outputs[677] = (inputs[677]-0.001000000047)/0.0316227749
		outputs[678] = (inputs[678]-0.001000000047)/0.0316227749
		outputs[679] = (inputs[679]-0.001000000047)/0.0316227749
		outputs[680] = (inputs[680]-0.001000000047)/0.0316227749
		outputs[681] = (inputs[681]-0.001000000047)/0.0316227749
		outputs[682] = (inputs[682]-0.001000000047)/0.0316227749
		outputs[683] = (inputs[683]-0.001000000047)/0.0316227749
		outputs[684] = (inputs[684]-0.001000000047)/0.0316227749
		outputs[685] = (inputs[685]-0.001000000047)/0.0316227749
		outputs[686] = (inputs[686]-0.001000000047)/0.0316227749
		outputs[687] = (inputs[687]-0.001000000047)/0.0316227749
		outputs[688] = (inputs[688]-0.001000000047)/0.0316227749
		outputs[689] = (inputs[689]-0.001000000047)/0.0316227749
		outputs[690] = (inputs[690]-0.001000000047)/0.0316227749
		outputs[691] = (inputs[691]-0.001000000047)/0.0316227749
		outputs[692] = (inputs[692]-0.001000000047)/0.0316227749
		outputs[693] = (inputs[693]-0.001000000047)/0.0316227749
		outputs[694] = (inputs[694]-0.001000000047)/0.0316227749
		outputs[695] = (inputs[695]-0.001000000047)/0.0316227749
		outputs[696] = (inputs[696]-0.001000000047)/0.0316227749
		outputs[697] = (inputs[697]-0.001000000047)/0.0316227749
		outputs[698] = (inputs[698]-0.001000000047)/0.0316227749
		outputs[699] = (inputs[699]-0.001000000047)/0.0316227749
		outputs[700] = (inputs[700]-0.001000000047)/0.0316227749
		outputs[701] = (inputs[701]-0.001000000047)/0.0316227749
		outputs[702] = (inputs[702]-0.001000000047)/0.0316227749
		outputs[703] = (inputs[703]-0.001000000047)/0.0316227749
		outputs[704] = (inputs[704]-0.001000000047)/0.0316227749
		outputs[705] = (inputs[705]-0.001000000047)/0.0316227749
		outputs[706] = (inputs[706]-0.001000000047)/0.0316227749
		outputs[707] = (inputs[707]-0.001000000047)/0.0316227749
		outputs[708] = (inputs[708]-0.001000000047)/0.0316227749
		outputs[709] = (inputs[709]-0.001000000047)/0.0316227749
		outputs[710] = (inputs[710]-0.001000000047)/0.0316227749
		outputs[711] = (inputs[711]-0.001000000047)/0.0316227749
		outputs[712] = (inputs[712]-0.001000000047)/0.0316227749
		outputs[713] = (inputs[713]-0.001000000047)/0.0316227749
		outputs[714] = (inputs[714]-0.001000000047)/0.0316227749
		outputs[715] = (inputs[715]-0.001000000047)/0.0316227749
		outputs[716] = (inputs[716]-0.001000000047)/0.0316227749
		outputs[717] = (inputs[717]-0.001000000047)/0.0316227749
		outputs[718] = (inputs[718]-0.001000000047)/0.0316227749
		outputs[719] = (inputs[719]-0.001000000047)/0.0316227749
		outputs[720] = (inputs[720]-0.001000000047)/0.0316227749
		outputs[721] = (inputs[721]-0.001000000047)/0.0316227749
		outputs[722] = (inputs[722]-0.001000000047)/0.0316227749
		outputs[723] = (inputs[723]-0.001000000047)/0.0316227749
		outputs[724] = (inputs[724]-0.001000000047)/0.0316227749
		outputs[725] = (inputs[725]-0.001000000047)/0.0316227749
		outputs[726] = (inputs[726]-0.001000000047)/0.0316227749
		outputs[727] = (inputs[727]-0.001000000047)/0.0316227749
		outputs[728] = (inputs[728]-0.001000000047)/0.0316227749
		outputs[729] = (inputs[729]-0.001000000047)/0.0316227749
		outputs[730] = (inputs[730]-0.001000000047)/0.0316227749
		outputs[731] = (inputs[731]-0.001000000047)/0.0316227749
		outputs[732] = (inputs[732]-0.001000000047)/0.0316227749
		outputs[733] = (inputs[733]-0.001000000047)/0.0316227749
		outputs[734] = (inputs[734]-0.001000000047)/0.0316227749
		outputs[735] = (inputs[735]-0.001000000047)/0.0316227749
		outputs[736] = (inputs[736]-0.001000000047)/0.0316227749
		outputs[737] = (inputs[737]-0.001000000047)/0.0316227749
		outputs[738] = (inputs[738]-0.001000000047)/0.0316227749
		outputs[739] = (inputs[739]-0.001000000047)/0.0316227749
		outputs[740] = (inputs[740]-0.001000000047)/0.0316227749
		outputs[741] = (inputs[741]-0.001000000047)/0.0316227749
		outputs[742] = (inputs[742]-0.001000000047)/0.0316227749
		outputs[743] = (inputs[743]-0.001000000047)/0.0316227749
		outputs[744] = (inputs[744]-0.001000000047)/0.0316227749
		outputs[745] = (inputs[745]-0.001000000047)/0.0316227749
		outputs[746] = (inputs[746]-0.001000000047)/0.0316227749
		outputs[747] = (inputs[747]-0.001000000047)/0.0316227749
		outputs[748] = (inputs[748]-0.001000000047)/0.0316227749
		outputs[749] = (inputs[749]-0.001000000047)/0.0316227749
		outputs[750] = (inputs[750]-0.001000000047)/0.0316227749
		outputs[751] = (inputs[751]-0.001000000047)/0.0316227749
		outputs[752] = (inputs[752]-0.001000000047)/0.0316227749
		outputs[753] = (inputs[753]-0.001000000047)/0.0316227749
		outputs[754] = (inputs[754]-0.001000000047)/0.0316227749
		outputs[755] = (inputs[755]-0.001000000047)/0.0316227749
		outputs[756] = (inputs[756]-0.001000000047)/0.0316227749
		outputs[757] = (inputs[757]-0.001000000047)/0.0316227749
		outputs[758] = (inputs[758]-0.001000000047)/0.0316227749
		outputs[759] = (inputs[759]-0.001000000047)/0.0316227749
		outputs[760] = (inputs[760]-0.001000000047)/0.0316227749
		outputs[761] = (inputs[761]-0.001000000047)/0.0316227749
		outputs[762] = (inputs[762]-0.001000000047)/0.0316227749
		outputs[763] = (inputs[763]-0.001000000047)/0.0316227749
		outputs[764] = (inputs[764]-0.001000000047)/0.0316227749
		outputs[765] = (inputs[765]-0.001000000047)/0.0316227749
		outputs[766] = (inputs[766]-0.001000000047)/0.0316227749
		outputs[767] = (inputs[767]-0.001000000047)/0.0316227749
		outputs[768] = (inputs[768]-0.001000000047)/0.0316227749
		outputs[769] = (inputs[769]-0.001000000047)/0.0316227749
		outputs[770] = (inputs[770]-0.001000000047)/0.0316227749
		outputs[771] = (inputs[771]-0.001000000047)/0.0316227749
		outputs[772] = (inputs[772]-0.001000000047)/0.0316227749
		outputs[773] = (inputs[773]-0.001000000047)/0.0316227749
		outputs[774] = (inputs[774]-0.001000000047)/0.0316227749
		outputs[775] = (inputs[775]-0.001000000047)/0.0316227749
		outputs[776] = (inputs[776]-0.001000000047)/0.0316227749
		outputs[777] = (inputs[777]-0.001000000047)/0.0316227749
		outputs[778] = (inputs[778]-0.001000000047)/0.0316227749
		outputs[779] = (inputs[779]-0.001000000047)/0.0316227749
		outputs[780] = (inputs[780]-0.001000000047)/0.0316227749
		outputs[781] = (inputs[781]-0.001000000047)/0.0316227749
		outputs[782] = (inputs[782]-0.001000000047)/0.0316227749
		outputs[783] = (inputs[783]-0.001000000047)/0.0316227749
		outputs[784] = (inputs[784]-0.001000000047)/0.0316227749
		outputs[785] = (inputs[785]-0.001000000047)/0.0316227749
		outputs[786] = (inputs[786]-0.001000000047)/0.0316227749
		outputs[787] = (inputs[787]-0.001000000047)/0.0316227749
		outputs[788] = (inputs[788]-0.001000000047)/0.0316227749
		outputs[789] = (inputs[789]-0.001000000047)/0.0316227749
		outputs[790] = (inputs[790]-0.001000000047)/0.0316227749
		outputs[791] = (inputs[791]-0.001000000047)/0.0316227749
		outputs[792] = (inputs[792]-0.001000000047)/0.0316227749
		outputs[793] = (inputs[793]-0.001000000047)/0.0316227749
		outputs[794] = (inputs[794]-0.001000000047)/0.0316227749
		outputs[795] = (inputs[795]-0.001000000047)/0.0316227749
		outputs[796] = (inputs[796]-0.001000000047)/0.0316227749
		outputs[797] = (inputs[797]-0.001000000047)/0.0316227749
		outputs[798] = (inputs[798]-0.001000000047)/0.0316227749
		outputs[799] = (inputs[799]-0.001000000047)/0.0316227749
		outputs[800] = (inputs[800]-0.001000000047)/0.0316227749
		outputs[801] = (inputs[801]-0.001000000047)/0.0316227749
		outputs[802] = (inputs[802]-0.001000000047)/0.0316227749
		outputs[803] = (inputs[803]-0.001000000047)/0.0316227749
		outputs[804] = (inputs[804]-0.001000000047)/0.0316227749
		outputs[805] = (inputs[805]-0.001000000047)/0.0316227749
		outputs[806] = (inputs[806]-0.001000000047)/0.0316227749
		outputs[807] = (inputs[807]-0.001000000047)/0.0316227749
		outputs[808] = (inputs[808]-0.001000000047)/0.0316227749
		outputs[809] = (inputs[809]-0.001000000047)/0.0316227749
		outputs[810] = (inputs[810]-0.001000000047)/0.0316227749
		outputs[811] = (inputs[811]-0.001000000047)/0.0316227749
		outputs[812] = (inputs[812]-0.001000000047)/0.0316227749
		outputs[813] = (inputs[813]-0.001000000047)/0.0316227749
		outputs[814] = (inputs[814]-0.001000000047)/0.0316227749
		outputs[815] = (inputs[815]-0.001000000047)/0.0316227749
		outputs[816] = (inputs[816]-0.001000000047)/0.0316227749
		outputs[817] = (inputs[817]-0.001000000047)/0.0316227749
		outputs[818] = (inputs[818]-0.001000000047)/0.0316227749
		outputs[819] = (inputs[819]-0.001000000047)/0.0316227749
		outputs[820] = (inputs[820]-0.001000000047)/0.0316227749
		outputs[821] = (inputs[821]-0.001000000047)/0.0316227749
		outputs[822] = (inputs[822]-0.001000000047)/0.0316227749
		outputs[823] = (inputs[823]-0.001000000047)/0.0316227749
		outputs[824] = (inputs[824]-0.001000000047)/0.0316227749
		outputs[825] = (inputs[825]-0.001000000047)/0.0316227749
		outputs[826] = (inputs[826]-0.001000000047)/0.0316227749
		outputs[827] = (inputs[827]-0.001000000047)/0.0316227749
		outputs[828] = (inputs[828]-0.001000000047)/0.0316227749
		outputs[829] = (inputs[829]-0.001000000047)/0.0316227749
		outputs[830] = (inputs[830]-0.001000000047)/0.0316227749
		outputs[831] = (inputs[831]-0.001000000047)/0.0316227749
		outputs[832] = (inputs[832]-0.001000000047)/0.0316227749
		outputs[833] = (inputs[833]-0.001000000047)/0.0316227749
		outputs[834] = (inputs[834]-0.001000000047)/0.0316227749
		outputs[835] = (inputs[835]-0.001000000047)/0.0316227749
		outputs[836] = (inputs[836]-0.001000000047)/0.0316227749
		outputs[837] = (inputs[837]-0.001000000047)/0.0316227749
		outputs[838] = (inputs[838]-0.001000000047)/0.0316227749
		outputs[839] = (inputs[839]-0.001000000047)/0.0316227749
		outputs[840] = (inputs[840]-0.001000000047)/0.0316227749
		outputs[841] = (inputs[841]-0.001000000047)/0.0316227749
		outputs[842] = (inputs[842]-0.001000000047)/0.0316227749
		outputs[843] = (inputs[843]-0.001000000047)/0.0316227749
		outputs[844] = (inputs[844]-0.001000000047)/0.0316227749
		outputs[845] = (inputs[845]-0.001000000047)/0.0316227749
		outputs[846] = (inputs[846]-0.001000000047)/0.0316227749
		outputs[847] = (inputs[847]-0.001000000047)/0.0316227749
		outputs[848] = (inputs[848]-0.001000000047)/0.0316227749
		outputs[849] = (inputs[849]-0.001000000047)/0.0316227749
		outputs[850] = (inputs[850]-0.001000000047)/0.0316227749
		outputs[851] = (inputs[851]-0.001000000047)/0.0316227749
		outputs[852] = (inputs[852]-0.001000000047)/0.0316227749
		outputs[853] = (inputs[853]-0.001000000047)/0.0316227749
		outputs[854] = (inputs[854]-0.001000000047)/0.0316227749
		outputs[855] = (inputs[855]-0.001000000047)/0.0316227749
		outputs[856] = (inputs[856]-0.001000000047)/0.0316227749
		outputs[857] = (inputs[857]-0.001000000047)/0.0316227749
		outputs[858] = (inputs[858]-0.001000000047)/0.0316227749
		outputs[859] = (inputs[859]-0.001000000047)/0.0316227749
		outputs[860] = (inputs[860]-0.001000000047)/0.0316227749
		outputs[861] = (inputs[861]-0.001000000047)/0.0316227749
		outputs[862] = (inputs[862]-0.001000000047)/0.0316227749
		outputs[863] = (inputs[863]-0.001000000047)/0.0316227749
		outputs[864] = (inputs[864]-0.001000000047)/0.0316227749
		outputs[865] = (inputs[865]-0.001000000047)/0.0316227749
		outputs[866] = (inputs[866]-0.001000000047)/0.0316227749
		outputs[867] = (inputs[867]-0.001000000047)/0.0316227749
		outputs[868] = (inputs[868]-0.001000000047)/0.0316227749
		outputs[869] = (inputs[869]-0.001000000047)/0.0316227749
		outputs[870] = (inputs[870]-0.001000000047)/0.0316227749
		outputs[871] = (inputs[871]-0.001000000047)/0.0316227749
		outputs[872] = (inputs[872]-0.001000000047)/0.0316227749
		outputs[873] = (inputs[873]-0.001000000047)/0.0316227749
		outputs[874] = (inputs[874]-0.001000000047)/0.0316227749
		outputs[875] = (inputs[875]-0.001000000047)/0.0316227749
		outputs[876] = (inputs[876]-0.001000000047)/0.0316227749
		outputs[877] = (inputs[877]-0.001000000047)/0.0316227749
		outputs[878] = (inputs[878]-0.001000000047)/0.0316227749
		outputs[879] = (inputs[879]-0.001000000047)/0.0316227749
		outputs[880] = (inputs[880]-0.001000000047)/0.0316227749
		outputs[881] = (inputs[881]-0.001000000047)/0.0316227749
		outputs[882] = (inputs[882]-0.001000000047)/0.0316227749
		outputs[883] = (inputs[883]-0.001000000047)/0.0316227749
		outputs[884] = (inputs[884]-0.001000000047)/0.0316227749
		outputs[885] = (inputs[885]-0.001000000047)/0.0316227749
		outputs[886] = (inputs[886]-0.001000000047)/0.0316227749
		outputs[887] = (inputs[887]-0.001000000047)/0.0316227749
		outputs[888] = (inputs[888]-0.001000000047)/0.0316227749
		outputs[889] = (inputs[889]-0.001000000047)/0.0316227749
		outputs[890] = (inputs[890]-0.001000000047)/0.0316227749
		outputs[891] = (inputs[891]-0.001000000047)/0.0316227749
		outputs[892] = (inputs[892]-0.001000000047)/0.0316227749
		outputs[893] = (inputs[893]-0.001000000047)/0.0316227749
		outputs[894] = (inputs[894]-0.001000000047)/0.0316227749
		outputs[895] = (inputs[895]-0.001000000047)/0.0316227749
		outputs[896] = (inputs[896]-0.001000000047)/0.0316227749
		outputs[897] = (inputs[897]-0.001000000047)/0.0316227749
		outputs[898] = (inputs[898]-0.001000000047)/0.0316227749
		outputs[899] = (inputs[899]-0.001000000047)/0.0316227749
		outputs[900] = (inputs[900]-0.001000000047)/0.0316227749
		outputs[901] = (inputs[901]-0.001000000047)/0.0316227749
		outputs[902] = (inputs[902]-0.001000000047)/0.0316227749
		outputs[903] = (inputs[903]-0.001000000047)/0.0316227749
		outputs[904] = (inputs[904]-0.001000000047)/0.0316227749
		outputs[905] = (inputs[905]-0.001000000047)/0.0316227749
		outputs[906] = (inputs[906]-0.001000000047)/0.0316227749
		outputs[907] = (inputs[907]-0.001000000047)/0.0316227749
		outputs[908] = (inputs[908]-0.001000000047)/0.0316227749
		outputs[909] = (inputs[909]-0.001000000047)/0.0316227749
		outputs[910] = (inputs[910]-0.001000000047)/0.0316227749
		outputs[911] = (inputs[911]-0.001000000047)/0.0316227749
		outputs[912] = (inputs[912]-0.001000000047)/0.0316227749
		outputs[913] = (inputs[913]-0.001000000047)/0.0316227749
		outputs[914] = (inputs[914]-0.001000000047)/0.0316227749
		outputs[915] = (inputs[915]-0.001000000047)/0.0316227749
		outputs[916] = (inputs[916]-0.001000000047)/0.0316227749
		outputs[917] = (inputs[917]-0.001000000047)/0.0316227749
		outputs[918] = (inputs[918]-0.001000000047)/0.0316227749
		outputs[919] = (inputs[919]-0.001000000047)/0.0316227749
		outputs[920] = (inputs[920]-0.001000000047)/0.0316227749
		outputs[921] = (inputs[921]-0.001000000047)/0.0316227749
		outputs[922] = (inputs[922]-0.001000000047)/0.0316227749
		outputs[923] = (inputs[923]-0.001000000047)/0.0316227749
		outputs[924] = (inputs[924]-0.001000000047)/0.0316227749
		outputs[925] = (inputs[925]-0.001000000047)/0.0316227749
		outputs[926] = (inputs[926]-0.001000000047)/0.0316227749
		outputs[927] = (inputs[927]-0.001000000047)/0.0316227749
		outputs[928] = (inputs[928]-0.001000000047)/0.0316227749
		outputs[929] = (inputs[929]-0.001000000047)/0.0316227749
		outputs[930] = (inputs[930]-0.001000000047)/0.0316227749
		outputs[931] = (inputs[931]-0.001000000047)/0.0316227749
		outputs[932] = (inputs[932]-0.001000000047)/0.0316227749
		outputs[933] = (inputs[933]-0.001000000047)/0.0316227749
		outputs[934] = (inputs[934]-0.001000000047)/0.0316227749
		outputs[935] = (inputs[935]-0.001000000047)/0.0316227749
		outputs[936] = (inputs[936]-0.001000000047)/0.0316227749
		outputs[937] = (inputs[937]-0.001000000047)/0.0316227749
		outputs[938] = (inputs[938]-0.001000000047)/0.0316227749
		outputs[939] = (inputs[939]-0.001000000047)/0.0316227749
		outputs[940] = (inputs[940]-0.001000000047)/0.0316227749
		outputs[941] = (inputs[941]-0.001000000047)/0.0316227749
		outputs[942] = (inputs[942]-0.001000000047)/0.0316227749
		outputs[943] = (inputs[943]-0.001000000047)/0.0316227749
		outputs[944] = (inputs[944]-0.001000000047)/0.0316227749
		outputs[945] = (inputs[945]-0.001000000047)/0.0316227749
		outputs[946] = (inputs[946]-0.001000000047)/0.0316227749
		outputs[947] = (inputs[947]-0.001000000047)/0.0316227749
		outputs[948] = (inputs[948]-0.001000000047)/0.0316227749
		outputs[949] = (inputs[949]-0.001000000047)/0.0316227749
		outputs[950] = (inputs[950]-0.001000000047)/0.0316227749
		outputs[951] = (inputs[951]-0.001000000047)/0.0316227749
		outputs[952] = (inputs[952]-0.001000000047)/0.0316227749
		outputs[953] = (inputs[953]-0.001000000047)/0.0316227749
		outputs[954] = (inputs[954]-0.001000000047)/0.0316227749
		outputs[955] = (inputs[955]-0.001000000047)/0.0316227749
		outputs[956] = (inputs[956]-0.001000000047)/0.0316227749
		outputs[957] = (inputs[957]-0.001000000047)/0.0316227749
		outputs[958] = (inputs[958]-0.001000000047)/0.0316227749
		outputs[959] = (inputs[959]-0.001000000047)/0.0316227749
		outputs[960] = (inputs[960]-0.001000000047)/0.0316227749
		outputs[961] = (inputs[961]-0.001000000047)/0.0316227749
		outputs[962] = (inputs[962]-0.001000000047)/0.0316227749
		outputs[963] = (inputs[963]-0.001000000047)/0.0316227749
		outputs[964] = (inputs[964]-0.001000000047)/0.0316227749
		outputs[965] = (inputs[965]-0.001000000047)/0.0316227749
		outputs[966] = (inputs[966]-0.001000000047)/0.0316227749
		outputs[967] = (inputs[967]-0.001000000047)/0.0316227749
		outputs[968] = (inputs[968]-0.001000000047)/0.0316227749
		outputs[969] = (inputs[969]-0.001000000047)/0.0316227749
		outputs[970] = (inputs[970]-0.001000000047)/0.0316227749
		outputs[971] = (inputs[971]-0.001000000047)/0.0316227749
		outputs[972] = (inputs[972]-0.001000000047)/0.0316227749
		outputs[973] = (inputs[973]-0.001000000047)/0.0316227749
		outputs[974] = (inputs[974]-0.001000000047)/0.0316227749
		outputs[975] = (inputs[975]-0.001000000047)/0.0316227749
		outputs[976] = (inputs[976]-0.001000000047)/0.0316227749
		outputs[977] = (inputs[977]-0.001000000047)/0.0316227749
		outputs[978] = (inputs[978]-0.001000000047)/0.0316227749
		outputs[979] = (inputs[979]-0.001000000047)/0.0316227749
		outputs[980] = (inputs[980]-0.001000000047)/0.0316227749
		outputs[981] = (inputs[981]-0.001000000047)/0.0316227749
		outputs[982] = (inputs[982]-0.001000000047)/0.0316227749
		outputs[983] = (inputs[983]-0.001000000047)/0.0316227749
		outputs[984] = (inputs[984]-0.001000000047)/0.0316227749
		outputs[985] = (inputs[985]-0.001000000047)/0.0316227749
		outputs[986] = (inputs[986]-0.001000000047)/0.0316227749
		outputs[987] = (inputs[987]-0.001000000047)/0.0316227749
		outputs[988] = (inputs[988]-0.001000000047)/0.0316227749
		outputs[989] = (inputs[989]-0.001000000047)/0.0316227749
		outputs[990] = (inputs[990]-0.001000000047)/0.0316227749
		outputs[991] = (inputs[991]-0.001000000047)/0.0316227749
		outputs[992] = (inputs[992]-0.001000000047)/0.0316227749
		outputs[993] = (inputs[993]-0.001000000047)/0.0316227749
		outputs[994] = (inputs[994]-0.001000000047)/0.0316227749
		outputs[995] = (inputs[995]-0.001000000047)/0.0316227749
		outputs[996] = (inputs[996]-0.001000000047)/0.0316227749
		outputs[997] = (inputs[997]-0.001000000047)/0.0316227749
		outputs[998] = (inputs[998]-0.001000000047)/0.0316227749
		outputs[999] = (inputs[999]-0.001000000047)/0.0316227749
		outputs[1000] = (inputs[1000]-0.001000000047)/0.0316227749
		outputs[1001] = (inputs[1001]-0.001000000047)/0.0316227749
		outputs[1002] = (inputs[1002]-0.001000000047)/0.0316227749
		outputs[1003] = (inputs[1003]-0.001000000047)/0.0316227749
		outputs[1004] = (inputs[1004]-0.001000000047)/0.0316227749
		outputs[1005] = (inputs[1005]-0.001000000047)/0.0316227749
		outputs[1006] = (inputs[1006]-0.001000000047)/0.0316227749
		outputs[1007] = (inputs[1007]-0.001000000047)/0.0316227749
		outputs[1008] = (inputs[1008]-0.001000000047)/0.0316227749
		outputs[1009] = (inputs[1009]-0.001000000047)/0.0316227749
		outputs[1010] = (inputs[1010]-0.001000000047)/0.0316227749
		outputs[1011] = (inputs[1011]-0.001000000047)/0.0316227749
		outputs[1012] = (inputs[1012]-0.001000000047)/0.0316227749
		outputs[1013] = (inputs[1013]-0.001000000047)/0.0316227749
		outputs[1014] = (inputs[1014]-0.001000000047)/0.0316227749
		outputs[1015] = (inputs[1015]-0.001000000047)/0.0316227749
		outputs[1016] = (inputs[1016]-0.001000000047)/0.0316227749
		outputs[1017] = (inputs[1017]-0.001000000047)/0.0316227749
		outputs[1018] = (inputs[1018]-0.001000000047)/0.0316227749
		outputs[1019] = (inputs[1019]-0.001000000047)/0.0316227749
		outputs[1020] = (inputs[1020]-0.001000000047)/0.0316227749
		outputs[1021] = (inputs[1021]-0.001000000047)/0.0316227749
		outputs[1022] = (inputs[1022]-0.001000000047)/0.0316227749
		outputs[1023] = (inputs[1023]-0.001000000047)/0.0316227749
		outputs[1024] = (inputs[1024]-0.001000000047)/0.0316227749
		outputs[1025] = (inputs[1025]-0.001000000047)/0.0316227749
		outputs[1026] = (inputs[1026]-0.001000000047)/0.0316227749
		outputs[1027] = (inputs[1027]-0.001000000047)/0.0316227749
		outputs[1028] = (inputs[1028]-0.001000000047)/0.0316227749
		outputs[1029] = (inputs[1029]-0.001000000047)/0.0316227749
		outputs[1030] = (inputs[1030]-0.001000000047)/0.0316227749
		outputs[1031] = (inputs[1031]-0.001000000047)/0.0316227749
		outputs[1032] = (inputs[1032]-0.001000000047)/0.0316227749
		outputs[1033] = (inputs[1033]-0.001000000047)/0.0316227749
		outputs[1034] = (inputs[1034]-0.001000000047)/0.0316227749
		outputs[1035] = (inputs[1035]-0.001000000047)/0.0316227749
		outputs[1036] = (inputs[1036]-0.001000000047)/0.0316227749
		outputs[1037] = (inputs[1037]-0.001000000047)/0.0316227749
		outputs[1038] = (inputs[1038]-0.001000000047)/0.0316227749
		outputs[1039] = (inputs[1039]-0.001000000047)/0.0316227749
		outputs[1040] = (inputs[1040]-0.001000000047)/0.0316227749
		outputs[1041] = (inputs[1041]-0.001000000047)/0.0316227749
		outputs[1042] = (inputs[1042]-0.001000000047)/0.0316227749
		outputs[1043] = (inputs[1043]-0.001000000047)/0.0316227749
		outputs[1044] = (inputs[1044]-0.001000000047)/0.0316227749
		outputs[1045] = (inputs[1045]-0.001000000047)/0.0316227749
		outputs[1046] = (inputs[1046]-0.001000000047)/0.0316227749
		outputs[1047] = (inputs[1047]-0.001000000047)/0.0316227749
		outputs[1048] = (inputs[1048]-0.001000000047)/0.0316227749
		outputs[1049] = (inputs[1049]-0.001000000047)/0.0316227749
		outputs[1050] = (inputs[1050]-0.001000000047)/0.0316227749
		outputs[1051] = (inputs[1051]-0.001000000047)/0.0316227749
		outputs[1052] = (inputs[1052]-0.001000000047)/0.0316227749
		outputs[1053] = (inputs[1053]-0.001000000047)/0.0316227749
		outputs[1054] = (inputs[1054]-0.001000000047)/0.0316227749
		outputs[1055] = (inputs[1055]-0.001000000047)/0.0316227749
		outputs[1056] = (inputs[1056]-0.001000000047)/0.0316227749
		outputs[1057] = (inputs[1057]-0.001000000047)/0.0316227749
		outputs[1058] = (inputs[1058]-0.001000000047)/0.0316227749
		outputs[1059] = (inputs[1059]-0.001000000047)/0.0316227749
		outputs[1060] = (inputs[1060]-0.001000000047)/0.0316227749
		outputs[1061] = (inputs[1061]-0.001000000047)/0.0316227749
		outputs[1062] = (inputs[1062]-0.001000000047)/0.0316227749
		outputs[1063] = (inputs[1063]-0.001000000047)/0.0316227749
		outputs[1064] = (inputs[1064]-0.001000000047)/0.0316227749
		outputs[1065] = (inputs[1065]-0.001000000047)/0.0316227749
		outputs[1066] = (inputs[1066]-0.001000000047)/0.0316227749
		outputs[1067] = (inputs[1067]-0.001000000047)/0.0316227749
		outputs[1068] = (inputs[1068]-0.001000000047)/0.0316227749
		outputs[1069] = (inputs[1069]-0.001000000047)/0.0316227749
		outputs[1070] = (inputs[1070]-0.001000000047)/0.0316227749
		outputs[1071] = (inputs[1071]-0.001000000047)/0.0316227749
		outputs[1072] = (inputs[1072]-0.001000000047)/0.0316227749
		outputs[1073] = (inputs[1073]-0.001000000047)/0.0316227749
		outputs[1074] = (inputs[1074]-0.001000000047)/0.0316227749
		outputs[1075] = (inputs[1075]-0.001000000047)/0.0316227749
		outputs[1076] = (inputs[1076]-0.001000000047)/0.0316227749
		outputs[1077] = (inputs[1077]-0.001000000047)/0.0316227749
		outputs[1078] = (inputs[1078]-0.001000000047)/0.0316227749
		outputs[1079] = (inputs[1079]-0.001000000047)/0.0316227749
		outputs[1080] = (inputs[1080]-0.001000000047)/0.0316227749
		outputs[1081] = (inputs[1081]-0.001000000047)/0.0316227749
		outputs[1082] = (inputs[1082]-0.001000000047)/0.0316227749
		outputs[1083] = (inputs[1083]-0.001000000047)/0.0316227749
		outputs[1084] = (inputs[1084]-0.001000000047)/0.0316227749
		outputs[1085] = (inputs[1085]-0.001000000047)/0.0316227749
		outputs[1086] = (inputs[1086]-0.001000000047)/0.0316227749
		outputs[1087] = (inputs[1087]-0.001000000047)/0.0316227749
		outputs[1088] = (inputs[1088]-0.001000000047)/0.0316227749
		outputs[1089] = (inputs[1089]-0.001000000047)/0.0316227749
		outputs[1090] = (inputs[1090]-0.001000000047)/0.0316227749
		outputs[1091] = (inputs[1091]-0.001000000047)/0.0316227749
		outputs[1092] = (inputs[1092]-0.001000000047)/0.0316227749
		outputs[1093] = (inputs[1093]-0.001000000047)/0.0316227749
		outputs[1094] = (inputs[1094]-0.001000000047)/0.0316227749
		outputs[1095] = (inputs[1095]-0.001000000047)/0.0316227749
		outputs[1096] = (inputs[1096]-0.001000000047)/0.0316227749
		outputs[1097] = (inputs[1097]-0.001000000047)/0.0316227749
		outputs[1098] = (inputs[1098]-0.001000000047)/0.0316227749
		outputs[1099] = (inputs[1099]-0.001000000047)/0.0316227749
		outputs[1100] = (inputs[1100]-0.001000000047)/0.0316227749
		outputs[1101] = (inputs[1101]-0.001000000047)/0.0316227749
		outputs[1102] = (inputs[1102]-0.001000000047)/0.0316227749
		outputs[1103] = (inputs[1103]-0.001000000047)/0.0316227749
		outputs[1104] = (inputs[1104]-0.001000000047)/0.0316227749
		outputs[1105] = (inputs[1105]-0.001000000047)/0.0316227749
		outputs[1106] = (inputs[1106]-0.001000000047)/0.0316227749
		outputs[1107] = (inputs[1107]-0.001000000047)/0.0316227749
		outputs[1108] = (inputs[1108]-0.001000000047)/0.0316227749
		outputs[1109] = (inputs[1109]-0.001000000047)/0.0316227749
		outputs[1110] = (inputs[1110]-0.001000000047)/0.0316227749
		outputs[1111] = (inputs[1111]-0.001000000047)/0.0316227749
		outputs[1112] = (inputs[1112]-0.001000000047)/0.0316227749
		outputs[1113] = (inputs[1113]-0.001000000047)/0.0316227749
		outputs[1114] = (inputs[1114]-0.001000000047)/0.0316227749
		outputs[1115] = (inputs[1115]-0.001000000047)/0.0316227749
		outputs[1116] = (inputs[1116]-0.001000000047)/0.0316227749
		outputs[1117] = (inputs[1117]-0.001000000047)/0.0316227749
		outputs[1118] = (inputs[1118]-0.001000000047)/0.0316227749
		outputs[1119] = (inputs[1119]-0.001000000047)/0.0316227749
		outputs[1120] = (inputs[1120]-0.001000000047)/0.0316227749
		outputs[1121] = (inputs[1121]-0.001000000047)/0.0316227749
		outputs[1122] = (inputs[1122]-0.001000000047)/0.0316227749
		outputs[1123] = (inputs[1123]-0.001000000047)/0.0316227749
		outputs[1124] = (inputs[1124]-0.001000000047)/0.0316227749
		outputs[1125] = (inputs[1125]-0.001000000047)/0.0316227749
		outputs[1126] = (inputs[1126]-0.001000000047)/0.0316227749
		outputs[1127] = (inputs[1127]-0.001000000047)/0.0316227749
		outputs[1128] = (inputs[1128]-0.001000000047)/0.0316227749
		outputs[1129] = (inputs[1129]-0.001000000047)/0.0316227749
		outputs[1130] = (inputs[1130]-0.001000000047)/0.0316227749
		outputs[1131] = (inputs[1131]-0.001000000047)/0.0316227749
		outputs[1132] = (inputs[1132]-0.001000000047)/0.0316227749
		outputs[1133] = (inputs[1133]-0.001000000047)/0.0316227749
		outputs[1134] = (inputs[1134]-0.001000000047)/0.0316227749
		outputs[1135] = (inputs[1135]-0.001000000047)/0.0316227749
		outputs[1136] = (inputs[1136]-0.001000000047)/0.0316227749
		outputs[1137] = (inputs[1137]-0.001000000047)/0.0316227749
		outputs[1138] = (inputs[1138]-0.001000000047)/0.0316227749
		outputs[1139] = (inputs[1139]-0.001000000047)/0.0316227749
		outputs[1140] = (inputs[1140]-0.001000000047)/0.0316227749
		outputs[1141] = (inputs[1141]-0.001000000047)/0.0316227749
		outputs[1142] = (inputs[1142]-0.001000000047)/0.0316227749
		outputs[1143] = (inputs[1143]-0.001000000047)/0.0316227749
		outputs[1144] = (inputs[1144]-0.001000000047)/0.0316227749
		outputs[1145] = (inputs[1145]-0.001000000047)/0.0316227749
		outputs[1146] = (inputs[1146]-0.001000000047)/0.0316227749
		outputs[1147] = (inputs[1147]-0.001000000047)/0.0316227749
		outputs[1148] = (inputs[1148]-0.001000000047)/0.0316227749
		outputs[1149] = (inputs[1149]-0.001000000047)/0.0316227749
		outputs[1150] = (inputs[1150]-0.001000000047)/0.0316227749
		outputs[1151] = (inputs[1151]-0.001000000047)/0.0316227749
		outputs[1152] = (inputs[1152]-0.001000000047)/0.0316227749
		outputs[1153] = (inputs[1153]-0.001000000047)/0.0316227749
		outputs[1154] = (inputs[1154]-0.001000000047)/0.0316227749
		outputs[1155] = (inputs[1155]-0.001000000047)/0.0316227749
		outputs[1156] = (inputs[1156]-0.001000000047)/0.0316227749
		outputs[1157] = (inputs[1157]-0.001000000047)/0.0316227749
		outputs[1158] = (inputs[1158]-0.001000000047)/0.0316227749
		outputs[1159] = (inputs[1159]-0.001000000047)/0.0316227749
		outputs[1160] = (inputs[1160]-0.001000000047)/0.0316227749
		outputs[1161] = (inputs[1161]-0.001000000047)/0.0316227749
		outputs[1162] = (inputs[1162]-0.001000000047)/0.0316227749
		outputs[1163] = (inputs[1163]-0.001000000047)/0.0316227749
		outputs[1164] = (inputs[1164]-0.001000000047)/0.0316227749
		outputs[1165] = (inputs[1165]-0.001000000047)/0.0316227749
		outputs[1166] = (inputs[1166]-0.001000000047)/0.0316227749
		outputs[1167] = (inputs[1167]-0.001000000047)/0.0316227749
		outputs[1168] = (inputs[1168]-0.001000000047)/0.0316227749
		outputs[1169] = (inputs[1169]-0.001000000047)/0.0316227749
		outputs[1170] = (inputs[1170]-0.001000000047)/0.0316227749
		outputs[1171] = (inputs[1171]-0.001000000047)/0.0316227749
		outputs[1172] = (inputs[1172]-0.001000000047)/0.0316227749
		outputs[1173] = (inputs[1173]-0.001000000047)/0.0316227749
		outputs[1174] = (inputs[1174]-0.001000000047)/0.0316227749
		outputs[1175] = (inputs[1175]-0.001000000047)/0.0316227749
		outputs[1176] = (inputs[1176]-0.001000000047)/0.0316227749
		outputs[1177] = (inputs[1177]-0.001000000047)/0.0316227749
		outputs[1178] = (inputs[1178]-0.001000000047)/0.0316227749
		outputs[1179] = (inputs[1179]-0.001000000047)/0.0316227749
		outputs[1180] = (inputs[1180]-0.001000000047)/0.0316227749
		outputs[1181] = (inputs[1181]-0.001000000047)/0.0316227749
		outputs[1182] = (inputs[1182]-0.001000000047)/0.0316227749
		outputs[1183] = (inputs[1183]-0.001000000047)/0.0316227749
		outputs[1184] = (inputs[1184]-0.001000000047)/0.0316227749
		outputs[1185] = (inputs[1185]-0.001000000047)/0.0316227749
		outputs[1186] = (inputs[1186]-0.001000000047)/0.0316227749
		outputs[1187] = (inputs[1187]-0.001000000047)/0.0316227749
		outputs[1188] = (inputs[1188]-0.001000000047)/0.0316227749
		outputs[1189] = (inputs[1189]-0.001000000047)/0.0316227749
		outputs[1190] = (inputs[1190]-0.001000000047)/0.0316227749
		outputs[1191] = (inputs[1191]-0.001000000047)/0.0316227749
		outputs[1192] = (inputs[1192]-0.001000000047)/0.0316227749
		outputs[1193] = (inputs[1193]-0.001000000047)/0.0316227749
		outputs[1194] = (inputs[1194]-0.001000000047)/0.0316227749
		outputs[1195] = (inputs[1195]-0.001000000047)/0.0316227749
		outputs[1196] = (inputs[1196]-0.001000000047)/0.0316227749
		outputs[1197] = (inputs[1197]-0.001000000047)/0.0316227749
		outputs[1198] = (inputs[1198]-0.001000000047)/0.0316227749
		outputs[1199] = (inputs[1199]-0.001000000047)/0.0316227749
		outputs[1200] = (inputs[1200]-0.001000000047)/0.0316227749
		outputs[1201] = (inputs[1201]-0.001000000047)/0.0316227749
		outputs[1202] = (inputs[1202]-0.001000000047)/0.0316227749
		outputs[1203] = (inputs[1203]-0.001000000047)/0.0316227749
		outputs[1204] = (inputs[1204]-0.001000000047)/0.0316227749
		outputs[1205] = (inputs[1205]-0.001000000047)/0.0316227749
		outputs[1206] = (inputs[1206]-0.001000000047)/0.0316227749
		outputs[1207] = (inputs[1207]-0.001000000047)/0.0316227749
		outputs[1208] = (inputs[1208]-0.001000000047)/0.0316227749
		outputs[1209] = (inputs[1209]-0.001000000047)/0.0316227749
		outputs[1210] = (inputs[1210]-0.001000000047)/0.0316227749
		outputs[1211] = (inputs[1211]-0.001000000047)/0.0316227749
		outputs[1212] = (inputs[1212]-0.001000000047)/0.0316227749
		outputs[1213] = (inputs[1213]-0.001000000047)/0.0316227749
		outputs[1214] = (inputs[1214]-0.001000000047)/0.0316227749
		outputs[1215] = (inputs[1215]-0.001000000047)/0.0316227749
		outputs[1216] = (inputs[1216]-0.001000000047)/0.0316227749
		outputs[1217] = (inputs[1217]-0.001000000047)/0.0316227749
		outputs[1218] = (inputs[1218]-0.001000000047)/0.0316227749
		outputs[1219] = (inputs[1219]-0.001000000047)/0.0316227749
		outputs[1220] = (inputs[1220]-0.001000000047)/0.0316227749
		outputs[1221] = (inputs[1221]-0.001000000047)/0.0316227749
		outputs[1222] = (inputs[1222]-0.001000000047)/0.0316227749
		outputs[1223] = (inputs[1223]-0.001000000047)/0.0316227749
		outputs[1224] = (inputs[1224]-0.001000000047)/0.0316227749
		outputs[1225] = (inputs[1225]-0.001000000047)/0.0316227749
		outputs[1226] = (inputs[1226]-0.001000000047)/0.0316227749
		outputs[1227] = (inputs[1227]-0.001000000047)/0.0316227749
		outputs[1228] = (inputs[1228]-0.001000000047)/0.0316227749
		outputs[1229] = (inputs[1229]-0.001000000047)/0.0316227749
		outputs[1230] = (inputs[1230]-0.001000000047)/0.0316227749
		outputs[1231] = (inputs[1231]-0.001000000047)/0.0316227749
		outputs[1232] = (inputs[1232]-0.001000000047)/0.0316227749
		outputs[1233] = (inputs[1233]-0.001000000047)/0.0316227749
		outputs[1234] = (inputs[1234]-0.001000000047)/0.0316227749
		outputs[1235] = (inputs[1235]-0.001000000047)/0.0316227749
		outputs[1236] = (inputs[1236]-0.001000000047)/0.0316227749
		outputs[1237] = (inputs[1237]-0.001000000047)/0.0316227749
		outputs[1238] = (inputs[1238]-0.001000000047)/0.0316227749
		outputs[1239] = (inputs[1239]-0.001000000047)/0.0316227749
		outputs[1240] = (inputs[1240]-0.001000000047)/0.0316227749
		outputs[1241] = (inputs[1241]-0.001000000047)/0.0316227749
		outputs[1242] = (inputs[1242]-0.001000000047)/0.0316227749
		outputs[1243] = (inputs[1243]-0.001000000047)/0.0316227749
		outputs[1244] = (inputs[1244]-0.001000000047)/0.0316227749
		outputs[1245] = (inputs[1245]-0.001000000047)/0.0316227749
		outputs[1246] = (inputs[1246]-0.001000000047)/0.0316227749
		outputs[1247] = (inputs[1247]-0.001000000047)/0.0316227749
		outputs[1248] = (inputs[1248]-0.001000000047)/0.0316227749
		outputs[1249] = (inputs[1249]-0.001000000047)/0.0316227749
		outputs[1250] = (inputs[1250]-0.001000000047)/0.0316227749
		outputs[1251] = (inputs[1251]-0.001000000047)/0.0316227749
		outputs[1252] = (inputs[1252]-0.001000000047)/0.0316227749
		outputs[1253] = (inputs[1253]-0.001000000047)/0.0316227749
		outputs[1254] = (inputs[1254]-0.001000000047)/0.0316227749
		outputs[1255] = (inputs[1255]-0.001000000047)/0.0316227749
		outputs[1256] = (inputs[1256]-0.001000000047)/0.0316227749
		outputs[1257] = (inputs[1257]-0.001000000047)/0.0316227749
		outputs[1258] = (inputs[1258]-0.001000000047)/0.0316227749
		outputs[1259] = (inputs[1259]-0.001000000047)/0.0316227749
		outputs[1260] = (inputs[1260]-0.001000000047)/0.0316227749
		outputs[1261] = (inputs[1261]-0.001000000047)/0.0316227749
		outputs[1262] = (inputs[1262]-0.001000000047)/0.0316227749
		outputs[1263] = (inputs[1263]-0.001000000047)/0.0316227749
		outputs[1264] = (inputs[1264]-0.001000000047)/0.0316227749
		outputs[1265] = (inputs[1265]-0.001000000047)/0.0316227749
		outputs[1266] = (inputs[1266]-0.001000000047)/0.0316227749
		outputs[1267] = (inputs[1267]-0.001000000047)/0.0316227749
		outputs[1268] = (inputs[1268]-0.001000000047)/0.0316227749
		outputs[1269] = (inputs[1269]-0.001000000047)/0.0316227749
		outputs[1270] = (inputs[1270]-0.001000000047)/0.0316227749
		outputs[1271] = (inputs[1271]-0.001000000047)/0.0316227749
		outputs[1272] = (inputs[1272]-0.001000000047)/0.0316227749
		outputs[1273] = (inputs[1273]-0.001000000047)/0.0316227749
		outputs[1274] = (inputs[1274]-0.001000000047)/0.0316227749
		outputs[1275] = (inputs[1275]-0.001000000047)/0.0316227749
		outputs[1276] = (inputs[1276]-0.001000000047)/0.0316227749
		outputs[1277] = (inputs[1277]-0.001000000047)/0.0316227749
		outputs[1278] = (inputs[1278]-0.001000000047)/0.0316227749
		outputs[1279] = (inputs[1279]-0.001000000047)/0.0316227749
		outputs[1280] = (inputs[1280]-0.001000000047)/0.0316227749
		outputs[1281] = (inputs[1281]-0.001000000047)/0.0316227749
		outputs[1282] = (inputs[1282]-0.001000000047)/0.0316227749
		outputs[1283] = (inputs[1283]-0.001000000047)/0.0316227749
		outputs[1284] = (inputs[1284]-0.001000000047)/0.0316227749
		outputs[1285] = (inputs[1285]-0.001000000047)/0.0316227749
		outputs[1286] = (inputs[1286]-0.001000000047)/0.0316227749
		outputs[1287] = (inputs[1287]-0.001000000047)/0.0316227749
		outputs[1288] = (inputs[1288]-0.001000000047)/0.0316227749
		outputs[1289] = (inputs[1289]-0.001000000047)/0.0316227749
		outputs[1290] = (inputs[1290]-0.001000000047)/0.0316227749
		outputs[1291] = (inputs[1291]-0.001000000047)/0.0316227749
		outputs[1292] = (inputs[1292]-0.001000000047)/0.0316227749
		outputs[1293] = (inputs[1293]-0.001000000047)/0.0316227749
		outputs[1294] = (inputs[1294]-0.001000000047)/0.0316227749
		outputs[1295] = (inputs[1295]-0.001000000047)/0.0316227749
		outputs[1296] = (inputs[1296]-0.001000000047)/0.0316227749
		outputs[1297] = (inputs[1297]-0.001000000047)/0.0316227749
		outputs[1298] = (inputs[1298]-0.001000000047)/0.0316227749
		outputs[1299] = (inputs[1299]-0.001000000047)/0.0316227749
		outputs[1300] = (inputs[1300]-0.001000000047)/0.0316227749
		outputs[1301] = (inputs[1301]-0.001000000047)/0.0316227749
		outputs[1302] = (inputs[1302]-0.001000000047)/0.0316227749
		outputs[1303] = (inputs[1303]-0.001000000047)/0.0316227749
		outputs[1304] = (inputs[1304]-0.001000000047)/0.0316227749
		outputs[1305] = (inputs[1305]-0.001000000047)/0.0316227749
		outputs[1306] = (inputs[1306]-0.001000000047)/0.0316227749
		outputs[1307] = (inputs[1307]-0.001000000047)/0.0316227749
		outputs[1308] = (inputs[1308]-0.001000000047)/0.0316227749
		outputs[1309] = (inputs[1309]-0.001000000047)/0.0316227749
		outputs[1310] = (inputs[1310]-0.001000000047)/0.0316227749
		outputs[1311] = (inputs[1311]-0.001000000047)/0.0316227749
		outputs[1312] = (inputs[1312]-0.001000000047)/0.0316227749
		outputs[1313] = (inputs[1313]-0.001000000047)/0.0316227749
		outputs[1314] = (inputs[1314]-0.001000000047)/0.0316227749
		outputs[1315] = (inputs[1315]-0.001000000047)/0.0316227749
		outputs[1316] = (inputs[1316]-0.001000000047)/0.0316227749
		outputs[1317] = (inputs[1317]-0.001000000047)/0.0316227749
		outputs[1318] = (inputs[1318]-0.001000000047)/0.0316227749
		outputs[1319] = (inputs[1319]-0.001000000047)/0.0316227749
		outputs[1320] = (inputs[1320]-0.001000000047)/0.0316227749
		outputs[1321] = (inputs[1321]-0.001000000047)/0.0316227749
		outputs[1322] = (inputs[1322]-0.001000000047)/0.0316227749
		outputs[1323] = (inputs[1323]-0.001000000047)/0.0316227749
		outputs[1324] = (inputs[1324]-0.001000000047)/0.0316227749
		outputs[1325] = (inputs[1325]-0.001000000047)/0.0316227749
		outputs[1326] = (inputs[1326]-0.001000000047)/0.0316227749
		outputs[1327] = (inputs[1327]-0.001000000047)/0.0316227749
		outputs[1328] = (inputs[1328]-0.001000000047)/0.0316227749
		outputs[1329] = (inputs[1329]-0.001000000047)/0.0316227749
		outputs[1330] = (inputs[1330]-0.001000000047)/0.0316227749
		outputs[1331] = (inputs[1331]-0.001000000047)/0.0316227749
		outputs[1332] = (inputs[1332]-0.001000000047)/0.0316227749
		outputs[1333] = (inputs[1333]-0.001000000047)/0.0316227749
		outputs[1334] = (inputs[1334]-0.001000000047)/0.0316227749
		outputs[1335] = (inputs[1335]-0.001000000047)/0.0316227749
		outputs[1336] = (inputs[1336]-0.001000000047)/0.0316227749
		outputs[1337] = (inputs[1337]-0.001000000047)/0.0316227749
		outputs[1338] = (inputs[1338]-0.001000000047)/0.0316227749
		outputs[1339] = (inputs[1339]-0.001000000047)/0.0316227749
		outputs[1340] = (inputs[1340]-0.001000000047)/0.0316227749
		outputs[1341] = (inputs[1341]-0.001000000047)/0.0316227749
		outputs[1342] = (inputs[1342]-0.001000000047)/0.0316227749
		outputs[1343] = (inputs[1343]-0.001000000047)/0.0316227749
		outputs[1344] = (inputs[1344]-0.001000000047)/0.0316227749
		outputs[1345] = (inputs[1345]-0.001000000047)/0.0316227749
		outputs[1346] = (inputs[1346]-0.001000000047)/0.0316227749
		outputs[1347] = (inputs[1347]-0.001000000047)/0.0316227749
		outputs[1348] = (inputs[1348]-0.001000000047)/0.0316227749
		outputs[1349] = (inputs[1349]-0.001000000047)/0.0316227749
		outputs[1350] = (inputs[1350]-0.001000000047)/0.0316227749
		outputs[1351] = (inputs[1351]-0.001000000047)/0.0316227749
		outputs[1352] = (inputs[1352]-0.001000000047)/0.0316227749
		outputs[1353] = (inputs[1353]-0.001000000047)/0.0316227749
		outputs[1354] = (inputs[1354]-0.001000000047)/0.0316227749
		outputs[1355] = (inputs[1355]-0.001000000047)/0.0316227749
		outputs[1356] = (inputs[1356]-0.001000000047)/0.0316227749
		outputs[1357] = (inputs[1357]-0.001000000047)/0.0316227749
		outputs[1358] = (inputs[1358]-0.001000000047)/0.0316227749
		outputs[1359] = (inputs[1359]-0.001000000047)/0.0316227749
		outputs[1360] = (inputs[1360]-0.001000000047)/0.0316227749
		outputs[1361] = (inputs[1361]-0.001000000047)/0.0316227749
		outputs[1362] = (inputs[1362]-0.001000000047)/0.0316227749
		outputs[1363] = (inputs[1363]-0.001000000047)/0.0316227749
		outputs[1364] = (inputs[1364]-0.001000000047)/0.0316227749
		outputs[1365] = (inputs[1365]-0.001000000047)/0.0316227749
		outputs[1366] = (inputs[1366]-0.001000000047)/0.0316227749
		outputs[1367] = (inputs[1367]-0.001000000047)/0.0316227749
		outputs[1368] = (inputs[1368]-0.001000000047)/0.0316227749
		outputs[1369] = (inputs[1369]-0.001000000047)/0.0316227749
		outputs[1370] = (inputs[1370]-0.001000000047)/0.0316227749
		outputs[1371] = (inputs[1371]-0.001000000047)/0.0316227749
		outputs[1372] = (inputs[1372]-0.001000000047)/0.0316227749
		outputs[1373] = (inputs[1373]-0.001000000047)/0.0316227749
		outputs[1374] = (inputs[1374]-0.001000000047)/0.0316227749
		outputs[1375] = (inputs[1375]-0.001000000047)/0.0316227749
		outputs[1376] = (inputs[1376]-0.001000000047)/0.0316227749
		outputs[1377] = (inputs[1377]-0.001000000047)/0.0316227749
		outputs[1378] = (inputs[1378]-0.001000000047)/0.0316227749
		outputs[1379] = (inputs[1379]-0.001000000047)/0.0316227749
		outputs[1380] = (inputs[1380]-0.001000000047)/0.0316227749
		outputs[1381] = (inputs[1381]-0.001000000047)/0.0316227749
		outputs[1382] = (inputs[1382]-0.001000000047)/0.0316227749
		outputs[1383] = (inputs[1383]-0.001000000047)/0.0316227749
		outputs[1384] = (inputs[1384]-0.001000000047)/0.0316227749
		outputs[1385] = (inputs[1385]-0.001000000047)/0.0316227749
		outputs[1386] = (inputs[1386]-0.001000000047)/0.0316227749
		outputs[1387] = (inputs[1387]-0.001000000047)/0.0316227749
		outputs[1388] = (inputs[1388]-0.001000000047)/0.0316227749
		outputs[1389] = (inputs[1389]-0.001000000047)/0.0316227749
		outputs[1390] = (inputs[1390]-0.001000000047)/0.0316227749
		outputs[1391] = (inputs[1391]-0.001000000047)/0.0316227749
		outputs[1392] = (inputs[1392]-0.001000000047)/0.0316227749
		outputs[1393] = (inputs[1393]-0.001000000047)/0.0316227749
		outputs[1394] = (inputs[1394]-0.001000000047)/0.0316227749
		outputs[1395] = (inputs[1395]-0.001000000047)/0.0316227749

		return outputs;


	def perceptron_layer_1(self,inputs):

		combinations = [None] * 6

		combinations[0] = 0.0131715 +0.0552377*inputs[0] +0.0574155*inputs[1] +0.231113*inputs[2] +0.159391*inputs[3] +0.0863773*inputs[4] +0.0771995*inputs[5] +0.044848*inputs[6] +0.00246923*inputs[7] +0.029292*inputs[8] -0.00104795*inputs[9] +0.00244308*inputs[10] +0.105387*inputs[11] +0.0260209*inputs[12] +0.109129*inputs[13] +0.0448722*inputs[14] +0.122598*inputs[15] -0.0036237*inputs[16] +0.0107693*inputs[17] +0.103868*inputs[18] +0.00164703*inputs[19] +0.0404104*inputs[20] -0.0848406*inputs[21] +0.0293704*inputs[22] +0.0324786*inputs[23] +0.124063*inputs[24] -0.00864076*inputs[25] -0.0660502*inputs[26] -0.00963549*inputs[27] -0.00977277*inputs[28] -0.0415675*inputs[29] -0.0092854*inputs[30] -0.0730183*inputs[31] -0.0362472*inputs[32] +0.0960091*inputs[33] -0.0533764*inputs[34] +0.108066*inputs[35] +0.0694646*inputs[36] +0.0398866*inputs[37] +0.0127145*inputs[38] -0.113344*inputs[39] -0.0577001*inputs[40] +0.128015*inputs[41] +0.057874*inputs[42] +0.0418216*inputs[43] -0.012069*inputs[44] +0.0153644*inputs[45] -0.0354622*inputs[46] +0.0735916*inputs[47] -0.0411425*inputs[48] -0.00445899*inputs[49] +0.102549*inputs[50] +0.0106184*inputs[51] -0.017774*inputs[52] -0.0165565*inputs[53] +0.00642238*inputs[54] -0.062095*inputs[55] -0.0259233*inputs[56] +0.0306044*inputs[57] -0.10889*inputs[58] -0.00498519*inputs[59] +0.0252626*inputs[60] -0.11622*inputs[61] -0.0379094*inputs[62] -0.0239653*inputs[63] -0.0418517*inputs[64] -0.0223055*inputs[65] +0.0707918*inputs[66] -0.0627152*inputs[67] +0.0219685*inputs[68] -0.0108511*inputs[69] -0.0115759*inputs[70] +0.0225147*inputs[71] -0.0024948*inputs[72] +0.0271554*inputs[73] -0.0866151*inputs[74] +0.00705693*inputs[75] +0.0258328*inputs[76] -0.0364453*inputs[77] -0.179847*inputs[78] +0.023995*inputs[79] -0.00623742*inputs[80] +0.0133608*inputs[81] +0.0763442*inputs[82] -0.000636551*inputs[83] +0.0142105*inputs[84] -0.0270386*inputs[85] +0.0424388*inputs[86] -0.0430312*inputs[87] -0.025755*inputs[88] +0.0127542*inputs[89] -0.0121626*inputs[90] +0.0303309*inputs[91] -0.053918*inputs[92] -0.0656187*inputs[93] +0.0566473*inputs[94] -0.0643544*inputs[95] -0.0138607*inputs[96] -0.0396387*inputs[97] -0.00992447*inputs[98] +0.00739468*inputs[99] -0.00590205*inputs[100] -0.00577001*inputs[101] +0.00911994*inputs[102] -0.0637302*inputs[103] -0.022685*inputs[104] -6.12693e-05*inputs[105] +0.0167585*inputs[106] +0.0128493*inputs[107] +0.0158571*inputs[108] -0.00441168*inputs[109] +0.0490374*inputs[110] +0.0603116*inputs[111] +0.0245079*inputs[112] -0.00869231*inputs[113] +0.0740209*inputs[114] -0.0404887*inputs[115] -0.0526501*inputs[116] +0.00188619*inputs[117] +0.0458256*inputs[118] -0.0177207*inputs[119] -0.0165419*inputs[120] -0.00282929*inputs[121] -0.0572752*inputs[122] -0.00199254*inputs[123] -0.00879564*inputs[124] -0.0169015*inputs[125] -0.00468237*inputs[126] -0.0113425*inputs[127] -0.0655354*inputs[128] +0.024793*inputs[129] -0.00486834*inputs[130] +0.0377988*inputs[131] +0.0255107*inputs[132] +0.0888916*inputs[133] -0.0177225*inputs[134] -0.0233635*inputs[135] -0.00933203*inputs[136] -0.0469716*inputs[137] +0.00715177*inputs[138] +0.00923559*inputs[139] +0.00622639*inputs[140] +0.0278914*inputs[141] +0.0204317*inputs[142] -0.105618*inputs[143] +0.0105773*inputs[144] -0.0344013*inputs[145] -0.00835217*inputs[146] +0.00823192*inputs[147] -0.00721058*inputs[148] -0.00121041*inputs[149] -0.0160741*inputs[150] +0.0623627*inputs[151] -0.053972*inputs[152] -0.0382553*inputs[153] +0.0117361*inputs[154] -0.00156208*inputs[155] -0.0232612*inputs[156] -0.00859674*inputs[157] -0.0599789*inputs[158] +0.0153952*inputs[159] +0.0346581*inputs[160] -0.010954*inputs[161] -0.0178401*inputs[162] -0.00517753*inputs[163] -0.0348854*inputs[164] -0.00540907*inputs[165] -0.0101608*inputs[166] -0.0362904*inputs[167] -0.024698*inputs[168] +0.053615*inputs[169] -0.0116097*inputs[170] -0.0197781*inputs[171] -0.0303918*inputs[172] -0.0198135*inputs[173] -0.00418271*inputs[174] +0.00606915*inputs[175] -0.028303*inputs[176] +0.0118731*inputs[177] -0.032842*inputs[178] +0.0088523*inputs[179] -0.035568*inputs[180] +0.00381911*inputs[181] -0.0509092*inputs[182] -0.00650109*inputs[183] -0.08333*inputs[184] +0.0186286*inputs[185] -0.0375856*inputs[186] +0.0170969*inputs[187] +0.00638102*inputs[188] -0.0235571*inputs[189] +0.0729919*inputs[190] +0.0449958*inputs[191] +0.0108716*inputs[192] +0.0195696*inputs[193] -0.00410473*inputs[194] -0.00355105*inputs[195] -0.00411163*inputs[196] -0.00733615*inputs[197] +0.024436*inputs[198] +0.0362412*inputs[199] +0.00658069*inputs[200] +0.0541577*inputs[201] +0.0225113*inputs[202] -0.0151188*inputs[203] -0.00261884*inputs[204] +0.00690415*inputs[205] +0.0328509*inputs[206] -0.00217947*inputs[207] -0.0228546*inputs[208] +0.0520005*inputs[209] -0.0578656*inputs[210] +0.00812092*inputs[211] -0.000933323*inputs[212] +0.00862905*inputs[213] +0.0221442*inputs[214] -0.00311097*inputs[215] -0.0144032*inputs[216] +0.00611465*inputs[217] +0.0386905*inputs[218] -0.00291721*inputs[219] -0.0207238*inputs[220] +0.0177502*inputs[221] -0.00742493*inputs[222] -0.0469*inputs[223] +0.00916039*inputs[224] -0.0466168*inputs[225] +0.0162937*inputs[226] -0.0101373*inputs[227] -0.0271712*inputs[228] -0.061318*inputs[229] +0.0277832*inputs[230] -0.0136716*inputs[231] +0.00104295*inputs[232] -0.00582453*inputs[233] -0.00654609*inputs[234] -0.00461308*inputs[235] -0.0230364*inputs[236] +0.00438992*inputs[237] +0.016897*inputs[238] -0.0162949*inputs[239] +0.0300712*inputs[240] +0.0205016*inputs[241] -0.00716487*inputs[242] +0.0565499*inputs[243] +0.0205646*inputs[244] -0.0477422*inputs[245] +0.00999726*inputs[246] +0.0204554*inputs[247] -0.0236642*inputs[248] -0.0559557*inputs[249] -0.0138205*inputs[250] -0.0322983*inputs[251] +0.0144688*inputs[252] +0.0206624*inputs[253] +0.00745507*inputs[254] +0.0335063*inputs[255] +0.0422516*inputs[256] +0.0757032*inputs[257] +0.0164207*inputs[258] +0.0366113*inputs[259] -0.0209978*inputs[260] -0.0138109*inputs[261] -0.0515577*inputs[262] +0.0317971*inputs[263] +0.00890173*inputs[264] -0.0192348*inputs[265] -0.0178466*inputs[266] -0.0059378*inputs[267] +0.0140416*inputs[268] -0.014521*inputs[269] +0.0364789*inputs[270] +0.0635035*inputs[271] -0.00471944*inputs[272] +0.000585186*inputs[273] -0.00861523*inputs[274] +0.0305792*inputs[275] -0.0367182*inputs[276] -0.04354*inputs[277] -0.00330582*inputs[278] -0.0521035*inputs[279] +0.0814266*inputs[280] -0.00427107*inputs[281] +0.0058738*inputs[282] -0.00580697*inputs[283] +0.00325829*inputs[284] -0.017468*inputs[285] -0.0141336*inputs[286] +0.0651274*inputs[287] +0.00173656*inputs[288] -0.00876223*inputs[289] -0.000304643*inputs[290] -0.00458914*inputs[291] -0.00506579*inputs[292] -0.00311312*inputs[293] -0.00489996*inputs[294] +0.0162166*inputs[295] +0.0173255*inputs[296] -0.015968*inputs[297] -0.00149776*inputs[298] -0.00964502*inputs[299] -0.001071*inputs[300] -0.00846624*inputs[301] -0.00882547*inputs[302] +0.0165398*inputs[303] -0.00370209*inputs[304] +0.0062257*inputs[305] -0.0273398*inputs[306] -0.00724073*inputs[307] -0.00190118*inputs[308] -0.0133094*inputs[309] +0.000764802*inputs[310] -0.0168425*inputs[311] +0.0145357*inputs[312] -0.0178252*inputs[313] -0.0328879*inputs[314] +0.00333673*inputs[315] -0.0147926*inputs[316] -0.0239004*inputs[317] -0.00820518*inputs[318] -0.0316702*inputs[319] +0.0334652*inputs[320] +0.00713893*inputs[321] -0.000559123*inputs[322] +0.00672471*inputs[323] +0.0103897*inputs[324] -0.0257599*inputs[325] -0.00072224*inputs[326] -0.0104245*inputs[327] +0.0173618*inputs[328] -0.0133739*inputs[329] -0.0116205*inputs[330] -0.0143414*inputs[331] +0.0107792*inputs[332] -0.000506039*inputs[333] +0.026152*inputs[334] -0.0179247*inputs[335] -0.0169605*inputs[336] +0.0277671*inputs[337] -0.00820196*inputs[338] -0.00216238*inputs[339] +0.0180963*inputs[340] -0.0318266*inputs[341] -0.0275618*inputs[342] -0.0161989*inputs[343] -0.0149454*inputs[344] -0.00564387*inputs[345] -0.0151639*inputs[346] -0.0535617*inputs[347] -0.0313004*inputs[348] -0.0142997*inputs[349] +0.0193455*inputs[350] -0.00072224*inputs[351] -0.0100258*inputs[352] +0.0288894*inputs[353] -0.00155179*inputs[354] -0.00991819*inputs[355] +0.012592*inputs[356] +0.0119402*inputs[357] +0.0131264*inputs[358] +0.02331*inputs[359] -0.00828051*inputs[360] -0.00892451*inputs[361] -0.0107692*inputs[362] +0.0182161*inputs[363] -0.00331459*inputs[364] -0.0172419*inputs[365] +0.00709814*inputs[366] -0.00624684*inputs[367] -0.00877647*inputs[368] -0.024673*inputs[369] +0.0434639*inputs[370] -0.00453253*inputs[371] -0.00235444*inputs[372] -0.0649115*inputs[373] -0.0175424*inputs[374] -0.000722238*inputs[375] -0.0155235*inputs[376] +0.0375545*inputs[377] -0.0415544*inputs[378] -0.0291031*inputs[379] -0.0113922*inputs[380] -0.00492413*inputs[381] +0.00381117*inputs[382] -0.0173927*inputs[383] -0.0392679*inputs[384] +0.00922817*inputs[385] -0.0113722*inputs[386] -0.00419576*inputs[387] +0.000460682*inputs[388] +0.0143972*inputs[389] +0.00230179*inputs[390] -0.00072224*inputs[391] +0.0128188*inputs[392] +0.023272*inputs[393] -0.00072224*inputs[394] +0.0146523*inputs[395] +0.00315455*inputs[396] -0.0445309*inputs[397] -0.0321642*inputs[398] -0.0410993*inputs[399] -0.0157546*inputs[400] -0.00072224*inputs[401] -0.00694825*inputs[402] -0.013068*inputs[403] +0.00343975*inputs[404] +0.0494666*inputs[405] -0.0022148*inputs[406] +0.0143686*inputs[407] +0.0287103*inputs[408] +0.0225742*inputs[409] -0.0147593*inputs[410] +0.00981653*inputs[411] -0.0144688*inputs[412] +0.00314676*inputs[413] -0.00959777*inputs[414] +0.0230662*inputs[415] -0.039333*inputs[416] +0.0258748*inputs[417] -0.0248386*inputs[418] +0.00582707*inputs[419] -0.0117631*inputs[420] -0.0443903*inputs[421] -0.00613615*inputs[422] +0.0072303*inputs[423] -0.0028218*inputs[424] -0.00072224*inputs[425] +0.00472229*inputs[426] -0.0154744*inputs[427] +0.00397756*inputs[428] -0.00010322*inputs[429] -0.000722239*inputs[430] -0.0224458*inputs[431] +0.00910403*inputs[432] -0.00310955*inputs[433] +0.0151165*inputs[434] -0.00221026*inputs[435] +0.0164254*inputs[436] +0.012379*inputs[437] -0.0053772*inputs[438] +0.0183214*inputs[439] -0.0200193*inputs[440] -0.0296008*inputs[441] -0.00620651*inputs[442] -0.0184216*inputs[443] -0.000722238*inputs[444] -0.0181882*inputs[445] -0.00525722*inputs[446] -0.023142*inputs[447] -0.000115842*inputs[448] -0.000589417*inputs[449] +0.019696*inputs[450] +0.0347427*inputs[451] -0.00315952*inputs[452] -0.0324262*inputs[453] +0.00498604*inputs[454] -0.0267734*inputs[455] -0.0353642*inputs[456] -0.0137372*inputs[457] +0.0565702*inputs[458] -0.0495786*inputs[459] -0.029204*inputs[460] +0.00411465*inputs[461] -0.0034838*inputs[462] -0.0203991*inputs[463] -0.0353819*inputs[464] -0.0538311*inputs[465] +0.000328325*inputs[466] -0.0104275*inputs[467] -0.0131478*inputs[468] -0.00589414*inputs[469] -0.0160708*inputs[470] -0.0214771*inputs[471] +0.0107121*inputs[472] -0.00880249*inputs[473] +0.0179575*inputs[474] +0.0179575*inputs[475] -0.0133754*inputs[476] -0.0252331*inputs[477] +0.00997468*inputs[478] -0.0107478*inputs[479] +0.00517215*inputs[480] +0.0394463*inputs[481] -0.0245498*inputs[482] -0.0208248*inputs[483] -0.000589417*inputs[484] -0.00818845*inputs[485] -0.0243531*inputs[486] -0.0219701*inputs[487] -0.0104428*inputs[488] -0.0172949*inputs[489] +0.00798668*inputs[490] -0.0237246*inputs[491] -0.0115289*inputs[492] -0.0308691*inputs[493] -0.0107039*inputs[494] -0.00742429*inputs[495] -0.00742432*inputs[496] -0.0111645*inputs[497] +0.0680006*inputs[498] -0.00622819*inputs[499] -0.0208248*inputs[500] -0.00880249*inputs[501] -0.000589415*inputs[502] -0.000589417*inputs[503] -0.0267295*inputs[504] +0.0374596*inputs[505] +0.0196882*inputs[506] +0.0142334*inputs[507] -0.000589417*inputs[508] -0.00385885*inputs[509] +0.00198183*inputs[510] -0.0511749*inputs[511] -0.0179227*inputs[512] +0.0060754*inputs[513] -0.000589417*inputs[514] +0.0285049*inputs[515] -0.000589417*inputs[516] -0.0121949*inputs[517] -0.0137372*inputs[518] -0.000589416*inputs[519] +0.00198183*inputs[520] +0.00623488*inputs[521] -0.0180113*inputs[522] +0.0024019*inputs[523] -0.00554666*inputs[524] -0.00554667*inputs[525] +0.00708759*inputs[526] -0.0144195*inputs[527] -0.000589417*inputs[528] +0.0029507*inputs[529] -0.00844528*inputs[530] +0.0205406*inputs[531] +0.0107674*inputs[532] +0.010842*inputs[533] -0.0169517*inputs[534] -0.0254319*inputs[535] -0.0143577*inputs[536] -0.0254175*inputs[537] +0.010842*inputs[538] +0.0115537*inputs[539] -0.000589417*inputs[540] -0.0117363*inputs[541] -0.0130978*inputs[542] +0.00219294*inputs[543] -0.000589415*inputs[544] -0.00844528*inputs[545] -0.00142007*inputs[546] +0.00579002*inputs[547] +0.0200393*inputs[548] +0.00598099*inputs[549] +0.00267929*inputs[550] +0.0135115*inputs[551] -0.00380028*inputs[552] -0.0150726*inputs[553] +0.00261399*inputs[554] -0.000589417*inputs[555] +0.00754179*inputs[556] +0.00409082*inputs[557] +0.0129252*inputs[558] -0.0403607*inputs[559] +0.0021117*inputs[560] +0.00589017*inputs[561] +0.00952123*inputs[562] +0.0359016*inputs[563] +0.046977*inputs[564] -0.000589417*inputs[565] -0.00659106*inputs[566] -0.010933*inputs[567] -0.000589417*inputs[568] +0.0151314*inputs[569] +0.00935008*inputs[570] -0.000589414*inputs[571] -0.0121427*inputs[572] -0.0337872*inputs[573] -0.000589417*inputs[574] +0.0264216*inputs[575] -0.000589417*inputs[576] -0.0394601*inputs[577] +0.0135669*inputs[578] -0.0248466*inputs[579] -0.00493141*inputs[580] -0.00153253*inputs[581] -0.000589414*inputs[582] -0.00844528*inputs[583] -0.00583113*inputs[584] -0.0203371*inputs[585] +0.0242741*inputs[586] -0.000589417*inputs[587] +0.029008*inputs[588] -0.000589417*inputs[589] -0.00856115*inputs[590] -0.000416579*inputs[591] -0.000589416*inputs[592] -0.0675324*inputs[593] -0.000912129*inputs[594] +0.0274138*inputs[595] -0.00450184*inputs[596] -0.0314696*inputs[597] -0.00510648*inputs[598] -0.00916124*inputs[599] +0.0196758*inputs[600] +0.0355901*inputs[601] -0.0119742*inputs[602] -0.0115414*inputs[603] -0.018147*inputs[604] -0.000589414*inputs[605] +0.0047768*inputs[606] -0.00510648*inputs[607] +0.00838863*inputs[608] +0.0180117*inputs[609] -0.000589417*inputs[610] +0.0117391*inputs[611] +0.0316962*inputs[612] -0.00818844*inputs[613] -0.000416584*inputs[614] -0.00698227*inputs[615] +0.0018532*inputs[616] -0.0374119*inputs[617] -0.0140885*inputs[618] -0.00153102*inputs[619] +0.0219599*inputs[620] -0.00430836*inputs[621] -0.0130372*inputs[622] +0.0219599*inputs[623] -0.000894089*inputs[624] +0.0124659*inputs[625] -0.01953*inputs[626] -0.000589416*inputs[627] -0.000589417*inputs[628] -0.000589415*inputs[629] -0.0175356*inputs[630] -0.0171768*inputs[631] +0.00690589*inputs[632] -0.035558*inputs[633] -0.00964679*inputs[634] -0.00554666*inputs[635] -0.000589417*inputs[636] +0.0107646*inputs[637] -0.02222*inputs[638] -0.000589417*inputs[639] -0.0364087*inputs[640] +0.0150211*inputs[641] +0.00339594*inputs[642] +0.0255772*inputs[643] -0.0364338*inputs[644] -0.0175356*inputs[645] -0.0563556*inputs[646] -0.0120181*inputs[647] -0.000416576*inputs[648] +0.00887072*inputs[649] +0.0027556*inputs[650] +0.0285521*inputs[651] -0.00826362*inputs[652] +0.00887072*inputs[653] -0.00826351*inputs[654] -0.00261315*inputs[655] +0.00887072*inputs[656] -0.000416584*inputs[657] -0.000416576*inputs[658] +0.00275564*inputs[659] -0.000416584*inputs[660] -0.000416584*inputs[661] -0.00404942*inputs[662] -0.000416577*inputs[663] +0.00632466*inputs[664] +0.00325512*inputs[665] -0.0378794*inputs[666] -0.023259*inputs[667] -0.000416578*inputs[668] +0.0241966*inputs[669] -0.00443947*inputs[670] -0.0242827*inputs[671] -0.000416583*inputs[672] -0.00404941*inputs[673] +0.00898728*inputs[674] +0.0137751*inputs[675] +0.00325512*inputs[676] -0.000416582*inputs[677] -0.000416584*inputs[678] -0.000416584*inputs[679] +0.00325513*inputs[680] -0.00261315*inputs[681] -0.000416584*inputs[682] +0.000407136*inputs[683] -0.000416579*inputs[684] -0.0320653*inputs[685] +0.0120543*inputs[686] -0.000416584*inputs[687] +0.0120543*inputs[688] +0.0120543*inputs[689] -0.000416584*inputs[690] -0.0219615*inputs[691] -0.000416584*inputs[692] -0.000416582*inputs[693] -0.00443943*inputs[694] -0.00495517*inputs[695] -0.00495517*inputs[696] -0.00041658*inputs[697] +0.0308557*inputs[698] +0.00898728*inputs[699] -0.00495517*inputs[700] -0.00495517*inputs[701] +0.0148895*inputs[702] -0.000416584*inputs[703] -0.000416584*inputs[704] -0.000416584*inputs[705] -0.000416583*inputs[706] -0.000416584*inputs[707] +0.0238002*inputs[708] -0.000416579*inputs[709] -0.000416582*inputs[710] -0.000416584*inputs[711] -0.00041658*inputs[712] +0.0148895*inputs[713] +0.0148895*inputs[714] +0.0148895*inputs[715] +0.00775647*inputs[716] -0.000416584*inputs[717] +0.020642*inputs[718] +0.020642*inputs[719] -0.000416583*inputs[720] -0.000416581*inputs[721] -0.000416584*inputs[722] -0.000416581*inputs[723] -0.000416584*inputs[724] -0.00438337*inputs[725] +0.00619844*inputs[726] +0.0205575*inputs[727] -0.0147805*inputs[728] -0.0147805*inputs[729] -0.0147805*inputs[730] +0.0205575*inputs[731] +0.0109158*inputs[732] -0.000416584*inputs[733] -0.000416584*inputs[734] -0.000416584*inputs[735] +0.00458741*inputs[736] +0.00458741*inputs[737] +0.00458741*inputs[738] +0.00458741*inputs[739] -0.00711759*inputs[740] +0.0205575*inputs[741] -0.0228363*inputs[742] -0.000416584*inputs[743] -0.000416578*inputs[744] -0.000416584*inputs[745] -0.00475872*inputs[746] -0.000416579*inputs[747] +0.0106076*inputs[748] -0.000416584*inputs[749] -0.000416582*inputs[750] +0.0232264*inputs[751] +0.0232264*inputs[752] -0.000416581*inputs[753] -0.000416577*inputs[754] -0.000416583*inputs[755] +0.0115202*inputs[756] -0.000416584*inputs[757] +0.0371439*inputs[758] -0.00475871*inputs[759] -0.00475871*inputs[760] -0.0134948*inputs[761] -0.0134948*inputs[762] -0.0134948*inputs[763] +0.0109158*inputs[764] -0.000416581*inputs[765] -0.000416584*inputs[766] -0.000416584*inputs[767] -0.000416583*inputs[768] -0.000416584*inputs[769] -0.000932703*inputs[770] -0.000416584*inputs[771] +0.00904205*inputs[772] -0.0208886*inputs[773] +0.00904205*inputs[774] +0.00922961*inputs[775] -0.00503796*inputs[776] +0.00904205*inputs[777] -0.00503796*inputs[778] -0.00503796*inputs[779] +0.000407136*inputs[780] +0.00789849*inputs[781] -0.0344577*inputs[782] -0.000932697*inputs[783] -0.000416584*inputs[784] +0.0262942*inputs[785] -0.00483566*inputs[786] -0.00483564*inputs[787] -0.00483564*inputs[788] -0.00483564*inputs[789] -0.0160053*inputs[790] +0.00772744*inputs[791] -0.000416583*inputs[792] -0.000416583*inputs[793] -0.0187697*inputs[794] -0.000416583*inputs[795] -0.0120258*inputs[796] -0.0120258*inputs[797] -0.000416584*inputs[798] -0.000416584*inputs[799] -0.000416579*inputs[800] -0.0140664*inputs[801] -0.0140665*inputs[802] -0.000416579*inputs[803] -0.00443943*inputs[804] -0.000416583*inputs[805] -0.019001*inputs[806] +0.012766*inputs[807] +0.0329632*inputs[808] -0.000416584*inputs[809] +0.012766*inputs[810] -0.00742367*inputs[811] -0.00742367*inputs[812] -0.00742367*inputs[813] -0.00292955*inputs[814] -0.00292954*inputs[815] +0.00904205*inputs[816] -0.000416584*inputs[817] +0.00957642*inputs[818] -0.000416584*inputs[819] +0.00900418*inputs[820] +0.00900418*inputs[821] +0.0282459*inputs[822] -0.000416577*inputs[823] -0.000416584*inputs[824] +0.0324044*inputs[825] -0.0254319*inputs[826] -0.0254319*inputs[827] -0.000416583*inputs[828] -0.000416582*inputs[829] -0.000416577*inputs[830] -0.000416584*inputs[831] -0.000416584*inputs[832] +0.0139049*inputs[833] +0.0139049*inputs[834] +0.0139049*inputs[835] -0.000416584*inputs[836] -0.000416582*inputs[837] -0.000416583*inputs[838] -0.00193132*inputs[839] -0.00193132*inputs[840] -0.00193134*inputs[841] +0.0132525*inputs[842] -0.000416576*inputs[843] -0.000416584*inputs[844] +0.0290534*inputs[845] -0.0259424*inputs[846] -0.00988351*inputs[847] -0.00041658*inputs[848] +0.0115537*inputs[849] -0.0389208*inputs[850] +0.0071686*inputs[851] +0.0071686*inputs[852] +0.0071686*inputs[853] -0.000416582*inputs[854] +0.0194276*inputs[855] -0.000416584*inputs[856] -0.000416584*inputs[857] -0.000416583*inputs[858] -0.00369783*inputs[859] -0.00369783*inputs[860] -0.000416584*inputs[861] -0.0114419*inputs[862] -0.00574152*inputs[863] -0.011442*inputs[864] -0.000416581*inputs[865] -0.0276366*inputs[866] -0.00328237*inputs[867] -0.00328245*inputs[868] +0.00776291*inputs[869] -0.000416578*inputs[870] -0.000416581*inputs[871] -0.000416583*inputs[872] -0.0387039*inputs[873] -0.00041658*inputs[874] -0.000416582*inputs[875] +0.0131152*inputs[876] +0.0131152*inputs[877] -0.0199655*inputs[878] -0.015631*inputs[879] -0.0225134*inputs[880] -0.0225134*inputs[881] -0.000416578*inputs[882] -0.0292378*inputs[883] -0.0292378*inputs[884] -0.0213381*inputs[885] -0.0229641*inputs[886] -0.000416583*inputs[887] -0.000416584*inputs[888] -0.000416583*inputs[889] -0.000416584*inputs[890] +0.017403*inputs[891] -0.000416575*inputs[892] -0.000416584*inputs[893] +0.0157417*inputs[894] +0.00525085*inputs[895] -0.000416584*inputs[896] +0.00525084*inputs[897] -0.000416581*inputs[898] -0.000416584*inputs[899] -0.000416584*inputs[900] +0.0372753*inputs[901] +0.037144*inputs[902] -0.00041658*inputs[903] +0.0116311*inputs[904] +0.00573917*inputs[905] -0.000416584*inputs[906] -0.000416584*inputs[907] +0.023546*inputs[908] +0.023546*inputs[909] -0.000416579*inputs[910] -0.000416579*inputs[911] -0.000416579*inputs[912] -0.000416576*inputs[913] -0.000416584*inputs[914] -0.000416584*inputs[915] +0.0175688*inputs[916] +0.0175688*inputs[917] +0.0175688*inputs[918] -0.000416584*inputs[919] -0.000416578*inputs[920] -0.000416584*inputs[921] -0.0177668*inputs[922] -0.0177668*inputs[923] +0.00521675*inputs[924] +0.00521675*inputs[925] +0.00521675*inputs[926] -0.000416584*inputs[927] -0.000416584*inputs[928] -0.000416576*inputs[929] +0.0086008*inputs[930] +0.0086008*inputs[931] +0.0086008*inputs[932] +0.00382102*inputs[933] +0.00382102*inputs[934] -0.000416581*inputs[935] -0.0204744*inputs[936] -0.0204744*inputs[937] -0.000416584*inputs[938] -0.000416581*inputs[939] +0.00789849*inputs[940] +0.0238817*inputs[941] +0.0157665*inputs[942] +0.0306693*inputs[943] +0.00874236*inputs[944] +0.00874236*inputs[945] -0.000416583*inputs[946] -0.000416584*inputs[947] -0.00230568*inputs[948] -0.00230567*inputs[949] +0.0287422*inputs[950] -0.0268739*inputs[951] -0.000416584*inputs[952] -0.000416576*inputs[953] -0.000416583*inputs[954] -0.000416583*inputs[955] -0.000416582*inputs[956] -0.000416579*inputs[957] -0.000416584*inputs[958] -0.000416583*inputs[959] -0.00041658*inputs[960] -0.000416584*inputs[961] -0.00041658*inputs[962] -0.000416584*inputs[963] +0.0220239*inputs[964] +0.0186117*inputs[965] +0.0276145*inputs[966] +0.0184175*inputs[967] -0.000416583*inputs[968] -0.000416582*inputs[969] +0.00771558*inputs[970] +0.00771558*inputs[971] -0.0269751*inputs[972] -0.026975*inputs[973] -0.000416577*inputs[974] +0.0220239*inputs[975] +0.00351631*inputs[976] -0.026792*inputs[977] +0.00351631*inputs[978] -0.0265192*inputs[979] -0.0268607*inputs[980] -0.0109913*inputs[981] -0.000416584*inputs[982] -0.00918007*inputs[983] -0.00918004*inputs[984] +0.00898728*inputs[985] -0.00338467*inputs[986] -0.0115209*inputs[987] -0.0253194*inputs[988] -0.000416583*inputs[989] -0.000416584*inputs[990] -0.00822105*inputs[991] -0.00822105*inputs[992] -0.00822105*inputs[993] +0.0172296*inputs[994] -0.000416584*inputs[995] -0.000416582*inputs[996] -0.0138147*inputs[997] -0.0174109*inputs[998] -0.0068833*inputs[999] -0.000416581*inputs[1000] +0.0242279*inputs[1001] -0.000416575*inputs[1002] -0.0106081*inputs[1003] -0.0106081*inputs[1004] -0.000416577*inputs[1005] -0.000416584*inputs[1006] -0.000416584*inputs[1007] -0.000416579*inputs[1008] -0.000416578*inputs[1009] -0.0286763*inputs[1010] +0.0383692*inputs[1011] -0.026792*inputs[1012] -0.000416582*inputs[1013] +0.0120669*inputs[1014] -0.000416579*inputs[1015] -0.000416584*inputs[1016] +0.00813731*inputs[1017] +0.00813732*inputs[1018] -0.000416584*inputs[1019] -0.0290009*inputs[1020] -0.000416581*inputs[1021] +0.00535872*inputs[1022] -0.000416583*inputs[1023] +0.0120669*inputs[1024] -0.000416582*inputs[1025] +0.0120669*inputs[1026] -0.0158796*inputs[1027] -0.000416584*inputs[1028] +0.00761642*inputs[1029] -0.0290195*inputs[1030] -0.0163506*inputs[1031] -0.0163506*inputs[1032] +0.00761642*inputs[1033] -0.000416583*inputs[1034] -0.000416584*inputs[1035] +0.00761642*inputs[1036] +0.00772387*inputs[1037] -0.0108259*inputs[1038] -0.000416584*inputs[1039] -0.0220711*inputs[1040] -0.000416584*inputs[1041] -0.00700956*inputs[1042] -0.00700955*inputs[1043] +0.011193*inputs[1044] +0.011193*inputs[1045] +0.011193*inputs[1046] +0.00772387*inputs[1047] +0.00772387*inputs[1048] -0.00884559*inputs[1049] +0.0193437*inputs[1050] -0.0147135*inputs[1051] -0.0147135*inputs[1052] -0.000416584*inputs[1053] -0.0323635*inputs[1054] -0.000416584*inputs[1055] -0.000416583*inputs[1056] -0.041859*inputs[1057] -0.000416582*inputs[1058] -0.000416584*inputs[1059] -0.000416584*inputs[1060] -0.0111896*inputs[1061] +0.00641477*inputs[1062] -0.00594679*inputs[1063] -0.00594679*inputs[1064] -0.000416581*inputs[1065] -0.000416582*inputs[1066] -0.0181679*inputs[1067] -0.0181679*inputs[1068] -0.00711759*inputs[1069] -0.00177739*inputs[1070] +0.0270934*inputs[1071] -0.0111896*inputs[1072] -0.000416579*inputs[1073] -0.000416584*inputs[1074] -0.000416584*inputs[1075] -0.000416584*inputs[1076] -0.00381827*inputs[1077] -0.00381827*inputs[1078] -0.00381827*inputs[1079] +0.00319073*inputs[1080] +0.00319073*inputs[1081] -0.000416584*inputs[1082] -0.000416584*inputs[1083] -0.00808888*inputs[1084] -0.000416584*inputs[1085] -0.000416576*inputs[1086] -0.000416584*inputs[1087] -0.000416583*inputs[1088] -0.000416583*inputs[1089] +0.00761642*inputs[1090] +0.0305307*inputs[1091] -0.000416576*inputs[1092] +0.0141307*inputs[1093] -0.000416584*inputs[1094] -0.02437*inputs[1095] +0.0249986*inputs[1096] -0.000416584*inputs[1097] +0.0168007*inputs[1098] -0.000416581*inputs[1099] -0.0158973*inputs[1100] -0.0351588*inputs[1101] +0.0387923*inputs[1102] +0.00718014*inputs[1103] +0.00805819*inputs[1104] -0.0148256*inputs[1105] -0.000416584*inputs[1106] +0.00718015*inputs[1107] -0.000416582*inputs[1108] +0.0165569*inputs[1109] -0.000416584*inputs[1110] -0.0111579*inputs[1111] -0.000416584*inputs[1112] +0.0190982*inputs[1113] -0.00728126*inputs[1114] -0.0167411*inputs[1115] +0.0165569*inputs[1116] +0.0072908*inputs[1117] +0.0259698*inputs[1118] -0.000416583*inputs[1119] -0.000416584*inputs[1120] -0.00574696*inputs[1121] -0.0110745*inputs[1122] +0.0321074*inputs[1123] -0.000416584*inputs[1124] +0.00234094*inputs[1125] -0.00728125*inputs[1126] +0.0079492*inputs[1127] -0.000416581*inputs[1128] +0.0072908*inputs[1129] -0.000416582*inputs[1130] -0.000416581*inputs[1131] -0.000416577*inputs[1132] -0.000416584*inputs[1133] -0.000416584*inputs[1134] -0.00589801*inputs[1135] -0.000416584*inputs[1136] +0.0156324*inputs[1137] -0.0341884*inputs[1138] -0.00728126*inputs[1139] -0.000416578*inputs[1140] +0.00523808*inputs[1141] +0.00523808*inputs[1142] -0.000416584*inputs[1143] -0.00728126*inputs[1144] -0.000416576*inputs[1145] -0.000416584*inputs[1146] -0.000416579*inputs[1147] -0.000416584*inputs[1148] -0.000416578*inputs[1149] +0.0205982*inputs[1150] -0.000416579*inputs[1151] -0.000416584*inputs[1152] -0.000416584*inputs[1153] -0.00574696*inputs[1154] -0.00574695*inputs[1155] -0.000416583*inputs[1156] -0.01482*inputs[1157] +0.0109479*inputs[1158] +0.0109479*inputs[1159] -0.020207*inputs[1160] +0.0101781*inputs[1161] +0.0247024*inputs[1162] -0.000416584*inputs[1163] -0.000416578*inputs[1164] +0.0101781*inputs[1165] -0.0287087*inputs[1166] -0.000416583*inputs[1167] -0.000416579*inputs[1168] +0.0109479*inputs[1169] -0.00786325*inputs[1170] -0.000416582*inputs[1171] -0.0109467*inputs[1172] -0.000416576*inputs[1173] -0.00786325*inputs[1174] -0.000416584*inputs[1175] -0.000416577*inputs[1176] -0.000416577*inputs[1177] -0.01482*inputs[1178] -0.000416584*inputs[1179] -0.000416577*inputs[1180] +0.0115479*inputs[1181] -0.000416584*inputs[1182] -0.000416584*inputs[1183] -0.000416577*inputs[1184] -0.000416584*inputs[1185] -0.000416584*inputs[1186] -0.00728126*inputs[1187] -0.003576*inputs[1188] -0.003576*inputs[1189] -0.00916124*inputs[1190] +0.0115479*inputs[1191] +0.0115479*inputs[1192] -0.0143228*inputs[1193] +0.0115479*inputs[1194] -0.000416583*inputs[1195] -0.000416584*inputs[1196] -0.00916124*inputs[1197] -0.0167141*inputs[1198] -0.0167141*inputs[1199] +0.0341473*inputs[1200] -0.000416584*inputs[1201] -0.000416582*inputs[1202] -0.000416584*inputs[1203] -0.000416584*inputs[1204] +0.00764067*inputs[1205] -0.0116847*inputs[1206] -0.000416584*inputs[1207] -0.000416584*inputs[1208] -0.000416581*inputs[1209] -0.000416575*inputs[1210] -0.000416584*inputs[1211] +0.00228398*inputs[1212] -0.0249055*inputs[1213] +0.00228398*inputs[1214] -0.000416584*inputs[1215] -0.000416579*inputs[1216] -0.000416584*inputs[1217] +0.00764067*inputs[1218] -0.0265585*inputs[1219] -0.000416583*inputs[1220] +0.00791315*inputs[1221] +0.00791315*inputs[1222] +0.00791315*inputs[1223] +0.00791315*inputs[1224] -0.00997591*inputs[1225] -0.000416584*inputs[1226] -0.000416584*inputs[1227] +0.00328065*inputs[1228] -0.000416584*inputs[1229] -0.00839333*inputs[1230] -0.00431598*inputs[1231] -0.00431598*inputs[1232] +0.00411145*inputs[1233] +0.00411145*inputs[1234] +0.00310999*inputs[1235] -0.000372956*inputs[1236] -0.000416584*inputs[1237] -0.00037295*inputs[1238] -0.000372951*inputs[1239] +0.00310999*inputs[1240] +0.00328065*inputs[1241] +0.0118217*inputs[1242] -0.000416584*inputs[1243] +0.0184347*inputs[1244] +0.0184347*inputs[1245] -0.000416577*inputs[1246] -0.000416584*inputs[1247] +0.0177972*inputs[1248] +0.0177972*inputs[1249] -0.000416584*inputs[1250] -0.000416581*inputs[1251] -0.000416584*inputs[1252] -0.001867*inputs[1253] -0.000416584*inputs[1254] -0.000416584*inputs[1255] -0.000416583*inputs[1256] -0.000416582*inputs[1257] -0.00655395*inputs[1258] -0.000416581*inputs[1259] -0.00655395*inputs[1260] -0.0302782*inputs[1261] -0.000416584*inputs[1262] -0.000416581*inputs[1263] -0.00186699*inputs[1264] -0.000416582*inputs[1265] -0.00186699*inputs[1266] -0.0102035*inputs[1267] -0.000416584*inputs[1268] +0.0170099*inputs[1269] -0.000416577*inputs[1270] -0.000416584*inputs[1271] -0.021235*inputs[1272] -0.000416584*inputs[1273] -0.000416584*inputs[1274] -0.000416583*inputs[1275] -0.000416584*inputs[1276] -0.00497645*inputs[1277] -0.000416583*inputs[1278] +0.0298314*inputs[1279] -0.000416584*inputs[1280] +0.0133438*inputs[1281] -0.000416584*inputs[1282] -0.000416579*inputs[1283] +0.016586*inputs[1284] -0.000416584*inputs[1285] -0.00497644*inputs[1286] -0.00497644*inputs[1287] -0.00497644*inputs[1288] -0.00041658*inputs[1289] -0.0372971*inputs[1290] -0.02833*inputs[1291] -0.000416584*inputs[1292] -0.021322*inputs[1293] -0.000416582*inputs[1294] +0.038621*inputs[1295] +0.000654537*inputs[1296] +0.000654533*inputs[1297] -0.000416584*inputs[1298] -0.00782583*inputs[1299] -0.00041658*inputs[1300] -0.00450779*inputs[1301] +0.0121453*inputs[1302] -0.00454733*inputs[1303] -0.000416582*inputs[1304] -0.000416581*inputs[1305] -0.0314951*inputs[1306] -0.000416584*inputs[1307] -0.000416584*inputs[1308] +0.00188803*inputs[1309] -0.000416581*inputs[1310] -0.00450779*inputs[1311] -0.00450779*inputs[1312] -0.00454733*inputs[1313] -0.00450779*inputs[1314] -0.00450779*inputs[1315] -0.000416581*inputs[1316] -0.000416584*inputs[1317] +0.00340146*inputs[1318] -0.000416584*inputs[1319] -0.000416582*inputs[1320] -0.000416582*inputs[1321] +0.0191292*inputs[1322] +0.0106127*inputs[1323] +0.0106127*inputs[1324] +0.0103429*inputs[1325] +0.00973707*inputs[1326] +0.00973707*inputs[1327] +0.00973707*inputs[1328] +0.00973707*inputs[1329] +0.0161932*inputs[1330] +0.0234721*inputs[1331] +0.00340692*inputs[1332] +0.00340692*inputs[1333] -0.000416584*inputs[1334] -0.000416584*inputs[1335] -0.000416583*inputs[1336] -0.0169805*inputs[1337] -0.000416582*inputs[1338] -0.00808888*inputs[1339] -0.0289474*inputs[1340] -0.000416584*inputs[1341] -0.00458337*inputs[1342] -0.00458337*inputs[1343] -0.000416582*inputs[1344] -0.000416584*inputs[1345] -0.000416584*inputs[1346] -0.000416584*inputs[1347] -0.00454733*inputs[1348] +0.00742766*inputs[1349] -0.0271892*inputs[1350] -0.0111187*inputs[1351] -0.0111187*inputs[1352] +0.00340147*inputs[1353] +0.0123854*inputs[1354] -0.00835643*inputs[1355] -0.00835639*inputs[1356] -0.00835642*inputs[1357] +0.00742766*inputs[1358] +0.00742766*inputs[1359] +0.0136824*inputs[1360] +0.00340146*inputs[1361] -0.000416584*inputs[1362] -0.00431598*inputs[1363] -0.00711759*inputs[1364] +0.0110769*inputs[1365] -0.00431598*inputs[1366] -0.000416577*inputs[1367] +0.0110769*inputs[1368] +0.0110769*inputs[1369] -0.000416584*inputs[1370] -0.00431598*inputs[1371] -0.000416584*inputs[1372] -0.00850179*inputs[1373] -0.000416584*inputs[1374] -0.0166907*inputs[1375] -0.0116787*inputs[1376] -0.00538016*inputs[1377] -0.00538017*inputs[1378] -0.00538017*inputs[1379] -0.000416584*inputs[1380] -0.0089*inputs[1381] -0.000416578*inputs[1382] -0.0373951*inputs[1383] -0.000416578*inputs[1384] -0.000416584*inputs[1385] -0.0085018*inputs[1386] -0.00838702*inputs[1387] -0.00838702*inputs[1388] -0.00838702*inputs[1389] -0.000416582*inputs[1390] -0.000416584*inputs[1391] -0.000416585*inputs[1392] +0.00340145*inputs[1393] -0.000416577*inputs[1394] +0.0201931*inputs[1395] 
		combinations[1] = 0.0133305 +0.0562125*inputs[0] +0.0579082*inputs[1] +0.234472*inputs[2] +0.161869*inputs[3] +0.0892244*inputs[4] +0.0784814*inputs[5] +0.0457439*inputs[6] +0.00264755*inputs[7] +0.029637*inputs[8] -0.000924123*inputs[9] +0.00252084*inputs[10] +0.107143*inputs[11] +0.0264447*inputs[12] +0.111222*inputs[13] +0.0456204*inputs[14] +0.124976*inputs[15] -0.0034684*inputs[16] +0.0109083*inputs[17] +0.105167*inputs[18] +0.00171992*inputs[19] +0.0409577*inputs[20] -0.0860272*inputs[21] +0.0290363*inputs[22] +0.0328644*inputs[23] +0.126049*inputs[24] -0.00872603*inputs[25] -0.0668827*inputs[26] -0.00973087*inputs[27] -0.00990006*inputs[28] -0.0420238*inputs[29] -0.00918608*inputs[30] -0.0739953*inputs[31] -0.0366457*inputs[32] +0.0972948*inputs[33] -0.0543572*inputs[34] +0.109632*inputs[35] +0.0702777*inputs[36] +0.0405871*inputs[37] +0.0131426*inputs[38] -0.114348*inputs[39] -0.0585348*inputs[40] +0.130633*inputs[41] +0.0597697*inputs[42] +0.0426378*inputs[43] -0.0120455*inputs[44] +0.0155558*inputs[45] -0.0358117*inputs[46] +0.0747634*inputs[47] -0.041444*inputs[48] -0.00451605*inputs[49] +0.104224*inputs[50] +0.0107186*inputs[51] -0.0179493*inputs[52] -0.016703*inputs[53] +0.00648668*inputs[54] -0.0625866*inputs[55] -0.0262637*inputs[56] +0.0310604*inputs[57] -0.110155*inputs[58] -0.00506087*inputs[59] +0.0244098*inputs[60] -0.117824*inputs[61] -0.0382354*inputs[62] -0.02422*inputs[63] -0.0421717*inputs[64] -0.0225057*inputs[65] +0.0721685*inputs[66] -0.0634298*inputs[67] +0.0222025*inputs[68] -0.0110268*inputs[69] -0.0117245*inputs[70] +0.0229669*inputs[71] -0.00252071*inputs[72] +0.0275536*inputs[73] -0.0876097*inputs[74] +0.00710512*inputs[75] +0.026324*inputs[76] -0.0367525*inputs[77] -0.182923*inputs[78] +0.0243316*inputs[79] -0.00637192*inputs[80] +0.0135307*inputs[81] +0.0777688*inputs[82] -0.000602475*inputs[83] +0.0144798*inputs[84] -0.0273302*inputs[85] +0.0428924*inputs[86] -0.0436903*inputs[87] -0.0260018*inputs[88] +0.0128869*inputs[89] -0.0122204*inputs[90] +0.0306801*inputs[91] -0.0546297*inputs[92] -0.0670228*inputs[93] +0.0574911*inputs[94] -0.0655124*inputs[95] -0.0140433*inputs[96] -0.0402389*inputs[97] -0.0100412*inputs[98] +0.00767735*inputs[99] -0.00595016*inputs[100] -0.00586964*inputs[101] +0.00904771*inputs[102] -0.0643716*inputs[103] -0.023129*inputs[104] -1.37715e-05*inputs[105] +0.0168658*inputs[106] +0.0129781*inputs[107] +0.0159644*inputs[108] -0.00447619*inputs[109] +0.0497091*inputs[110] +0.0610586*inputs[111] +0.0250348*inputs[112] -0.00870524*inputs[113] +0.0753369*inputs[114] -0.0409428*inputs[115] -0.0531604*inputs[116] +0.00191108*inputs[117] +0.0464165*inputs[118] -0.0178088*inputs[119] -0.0166669*inputs[120] -0.00268901*inputs[121] -0.0578751*inputs[122] -0.0020182*inputs[123] -0.00890581*inputs[124] -0.0170616*inputs[125] -0.00474752*inputs[126] -0.0114103*inputs[127] -0.0663773*inputs[128] +0.0250623*inputs[129] -0.004928*inputs[130] +0.0382035*inputs[131] +0.0257291*inputs[132] +0.0905211*inputs[133] -0.0180135*inputs[134] -0.023592*inputs[135] -0.00939496*inputs[136] -0.047383*inputs[137] +0.00722341*inputs[138] +0.00931814*inputs[139] +0.00629301*inputs[140] +0.0283842*inputs[141] +0.0206886*inputs[142] -0.107313*inputs[143] +0.0105488*inputs[144] -0.0347328*inputs[145] -0.00845896*inputs[146] +0.00835001*inputs[147] -0.00732658*inputs[148] -0.00127791*inputs[149] -0.0162104*inputs[150] +0.063275*inputs[151] -0.0545749*inputs[152] -0.0386074*inputs[153] +0.0120252*inputs[154] -0.00160434*inputs[155] -0.023458*inputs[156] -0.00864336*inputs[157] -0.0605999*inputs[158] +0.0154515*inputs[159] +0.0350375*inputs[160] -0.0110713*inputs[161] -0.0180034*inputs[162] -0.00520544*inputs[163] -0.035216*inputs[164] -0.00542404*inputs[165] -0.0101883*inputs[166] -0.0366479*inputs[167] -0.0249554*inputs[168] +0.0543111*inputs[169] -0.0116767*inputs[170] -0.0199965*inputs[171] -0.0307661*inputs[172] -0.0199972*inputs[173] -0.00412728*inputs[174] +0.00614864*inputs[175] -0.0284933*inputs[176] +0.0119799*inputs[177] -0.0331773*inputs[178] +0.00891337*inputs[179] -0.0360063*inputs[180] +0.00378083*inputs[181] -0.0515267*inputs[182] -0.00653289*inputs[183] -0.0845902*inputs[184] +0.0188152*inputs[185] -0.0380905*inputs[186] +0.0172263*inputs[187] +0.00647532*inputs[188] -0.0237418*inputs[189] +0.0741048*inputs[190] +0.0455526*inputs[191] +0.0109943*inputs[192] +0.0197579*inputs[193] -0.00418014*inputs[194] -0.00355201*inputs[195] -0.00410603*inputs[196] -0.00738543*inputs[197] +0.0247759*inputs[198] +0.0366694*inputs[199] +0.0066947*inputs[200] +0.0548816*inputs[201] +0.0228517*inputs[202] -0.0153423*inputs[203] -0.00263112*inputs[204] +0.00696457*inputs[205] +0.0332901*inputs[206] -0.00219729*inputs[207] -0.0230074*inputs[208] +0.0527269*inputs[209] -0.0584767*inputs[210] +0.00823481*inputs[211] -0.000944508*inputs[212] +0.00877374*inputs[213] +0.0226434*inputs[214] -0.00312859*inputs[215] -0.0145659*inputs[216] +0.00618919*inputs[217] +0.0390885*inputs[218] -0.00297358*inputs[219] -0.0210295*inputs[220] +0.0181557*inputs[221] -0.00745031*inputs[222] -0.0473959*inputs[223] +0.00929829*inputs[224] -0.0470835*inputs[225] +0.0165217*inputs[226] -0.0101947*inputs[227] -0.0273831*inputs[228] -0.062058*inputs[229] +0.0280469*inputs[230] -0.0137609*inputs[231] +0.00105847*inputs[232] -0.00585666*inputs[233] -0.00658348*inputs[234] -0.00461092*inputs[235] -0.0233198*inputs[236] +0.00442803*inputs[237] +0.0170185*inputs[238] -0.0164028*inputs[239] +0.0303112*inputs[240] +0.020792*inputs[241] -0.0072225*inputs[242] +0.0573605*inputs[243] +0.0207089*inputs[244] -0.0482147*inputs[245] +0.0101213*inputs[246] +0.0206851*inputs[247] -0.0238506*inputs[248] -0.056598*inputs[249] -0.0140634*inputs[250] -0.032781*inputs[251] +0.0145624*inputs[252] +0.0209139*inputs[253] +0.00757952*inputs[254] +0.0339083*inputs[255] +0.0427172*inputs[256] +0.0770091*inputs[257] +0.016549*inputs[258] +0.037068*inputs[259] -0.021063*inputs[260] -0.0139412*inputs[261] -0.0520973*inputs[262] +0.0322007*inputs[263] +0.00906353*inputs[264] -0.0193301*inputs[265] -0.0180836*inputs[266] -0.00597596*inputs[267] +0.0141912*inputs[268] -0.014609*inputs[269] +0.0369937*inputs[270] +0.06436*inputs[271] -0.00475084*inputs[272] +0.000612217*inputs[273] -0.00864754*inputs[274] +0.0308912*inputs[275] -0.0369945*inputs[276] -0.0440077*inputs[277] -0.00328797*inputs[278] -0.0526043*inputs[279] +0.0826691*inputs[280] -0.00422568*inputs[281] +0.0059594*inputs[282] -0.00582306*inputs[283] +0.00327081*inputs[284] -0.0176798*inputs[285] -0.0142976*inputs[286] +0.0659955*inputs[287] +0.0017279*inputs[288] -0.00881285*inputs[289] -0.000326201*inputs[290] -0.00459831*inputs[291] -0.00511451*inputs[292] -0.00312181*inputs[293] -0.00487997*inputs[294] +0.0164242*inputs[295] +0.0174722*inputs[296] -0.0160947*inputs[297] -0.00162126*inputs[298] -0.00968398*inputs[299] -0.00107188*inputs[300] -0.00854298*inputs[301] -0.00894026*inputs[302] +0.0166801*inputs[303] -0.00374064*inputs[304] +0.00627181*inputs[305] -0.027556*inputs[306] -0.00724933*inputs[307] -0.00194048*inputs[308] -0.0134957*inputs[309] +0.000743976*inputs[310] -0.0170006*inputs[311] +0.0147948*inputs[312] -0.0179209*inputs[313] -0.0333352*inputs[314] +0.00335906*inputs[315] -0.0149027*inputs[316] -0.0241277*inputs[317] -0.00833719*inputs[318] -0.0320155*inputs[319] +0.0338223*inputs[320] +0.00719079*inputs[321] -0.000565783*inputs[322] +0.00688662*inputs[323] +0.0104757*inputs[324] -0.0259902*inputs[325] -0.000730877*inputs[326] -0.0104855*inputs[327] +0.0174847*inputs[328] -0.0135462*inputs[329] -0.0116346*inputs[330] -0.0144247*inputs[331] +0.0108654*inputs[332] -0.000526445*inputs[333] +0.0264939*inputs[334] -0.0180539*inputs[335] -0.0170681*inputs[336] +0.0282107*inputs[337] -0.00825735*inputs[338] -0.00217936*inputs[339] +0.0183048*inputs[340] -0.0322833*inputs[341] -0.0277854*inputs[342] -0.0164014*inputs[343] -0.0150372*inputs[344] -0.00571194*inputs[345] -0.0152574*inputs[346] -0.0541742*inputs[347] -0.0316683*inputs[348] -0.0144721*inputs[349] +0.019491*inputs[350] -0.000730877*inputs[351] -0.01008*inputs[352] +0.0291876*inputs[353] -0.00154721*inputs[354] -0.00999607*inputs[355] +0.0126797*inputs[356] +0.012044*inputs[357] +0.013319*inputs[358] +0.0235911*inputs[359] -0.00835329*inputs[360] -0.00903711*inputs[361] -0.0108552*inputs[362] +0.0183447*inputs[363] -0.00334585*inputs[364] -0.0174554*inputs[365] +0.00716943*inputs[366] -0.00627673*inputs[367] -0.00880748*inputs[368] -0.024934*inputs[369] +0.0440694*inputs[370] -0.00455211*inputs[371] -0.00237646*inputs[372] -0.065797*inputs[373] -0.0176562*inputs[374] -0.000730877*inputs[375] -0.0156739*inputs[376] +0.0380133*inputs[377] -0.0420611*inputs[378] -0.0293574*inputs[379] -0.0114504*inputs[380] -0.00506077*inputs[381] +0.00383697*inputs[382] -0.0175124*inputs[383] -0.0396636*inputs[384] +0.00934476*inputs[385] -0.0114472*inputs[386] -0.00420372*inputs[387] +0.000491205*inputs[388] +0.0145332*inputs[389] +0.00234213*inputs[390] -0.000730877*inputs[391] +0.0129115*inputs[392] +0.0235065*inputs[393] -0.000730877*inputs[394] +0.0147876*inputs[395] +0.00317718*inputs[396] -0.0450988*inputs[397] -0.0323979*inputs[398] -0.041627*inputs[399] -0.0158785*inputs[400] -0.000730877*inputs[401] -0.00700182*inputs[402] -0.0131552*inputs[403] +0.0034819*inputs[404] +0.0500953*inputs[405] -0.00224836*inputs[406] +0.0144813*inputs[407] +0.0290321*inputs[408] +0.0228426*inputs[409] -0.0149212*inputs[410] +0.0098466*inputs[411] -0.0145631*inputs[412] +0.00319235*inputs[413] -0.0096458*inputs[414] +0.023256*inputs[415] -0.0399216*inputs[416] +0.026121*inputs[417] -0.0250686*inputs[418] +0.00586693*inputs[419] -0.0118067*inputs[420] -0.0450113*inputs[421] -0.00618719*inputs[422] +0.00728721*inputs[423] -0.00285123*inputs[424] -0.000730877*inputs[425] +0.0047043*inputs[426] -0.0156676*inputs[427] +0.00399192*inputs[428] -0.000123942*inputs[429] -0.000730877*inputs[430] -0.0226795*inputs[431] +0.00920856*inputs[432] -0.00311591*inputs[433] +0.0151785*inputs[434] -0.00224282*inputs[435] +0.0166105*inputs[436] +0.0124935*inputs[437] -0.00534572*inputs[438] +0.0184921*inputs[439] -0.020118*inputs[440] -0.0298278*inputs[441] -0.00626856*inputs[442] -0.0185667*inputs[443] -0.000730877*inputs[444] -0.0183316*inputs[445] -0.00528018*inputs[446] -0.0232556*inputs[447] -0.000107875*inputs[448] -0.000596464*inputs[449] +0.0199266*inputs[450] +0.0352613*inputs[451] -0.00319204*inputs[452] -0.0326909*inputs[453] +0.00502971*inputs[454] -0.0270767*inputs[455] -0.0356878*inputs[456] -0.0138405*inputs[457] +0.057327*inputs[458] -0.0501725*inputs[459] -0.0293862*inputs[460] +0.00418255*inputs[461] -0.00349596*inputs[462] -0.0207644*inputs[463] -0.0356942*inputs[464] -0.0546316*inputs[465] +0.000292961*inputs[466] -0.0105079*inputs[467] -0.013252*inputs[468] -0.00590004*inputs[469] -0.0161828*inputs[470] -0.0216446*inputs[471] +0.0107601*inputs[472] -0.00890287*inputs[473] +0.0181577*inputs[474] +0.0181577*inputs[475] -0.0134388*inputs[476] -0.0254454*inputs[477] +0.0101568*inputs[478] -0.0107739*inputs[479] +0.0051755*inputs[480] +0.0399575*inputs[481] -0.0248352*inputs[482] -0.0211115*inputs[483] -0.000596465*inputs[484] -0.00822287*inputs[485] -0.0246179*inputs[486] -0.0221421*inputs[487] -0.0105543*inputs[488] -0.0174178*inputs[489] +0.00811413*inputs[490] -0.0238822*inputs[491] -0.0116201*inputs[492] -0.0311252*inputs[493] -0.010795*inputs[494] -0.00746347*inputs[495] -0.00746347*inputs[496] -0.0112155*inputs[497] +0.0690256*inputs[498] -0.00626215*inputs[499] -0.0211115*inputs[500] -0.00890287*inputs[501] -0.000596465*inputs[502] -0.000596465*inputs[503] -0.0269858*inputs[504] +0.0379486*inputs[505] +0.0199318*inputs[506] +0.0144366*inputs[507] -0.000596465*inputs[508] -0.00385526*inputs[509] +0.00200039*inputs[510] -0.0517567*inputs[511] -0.0180748*inputs[512] +0.00605692*inputs[513] -0.000596465*inputs[514] +0.0288795*inputs[515] -0.000596465*inputs[516] -0.012412*inputs[517] -0.0138405*inputs[518] -0.000596465*inputs[519] +0.00200039*inputs[520] +0.0062739*inputs[521] -0.0181413*inputs[522] +0.00237634*inputs[523] -0.00557873*inputs[524] -0.00557873*inputs[525] +0.00724982*inputs[526] -0.0145189*inputs[527] -0.000596465*inputs[528] +0.00296117*inputs[529] -0.00850785*inputs[530] +0.0206527*inputs[531] +0.0107961*inputs[532] +0.0109416*inputs[533] -0.0170776*inputs[534] -0.0257271*inputs[535] -0.0144326*inputs[536] -0.0257031*inputs[537] +0.0109416*inputs[538] +0.0116246*inputs[539] -0.000596465*inputs[540] -0.0118595*inputs[541] -0.0132387*inputs[542] +0.00217869*inputs[543] -0.000596464*inputs[544] -0.00850785*inputs[545] -0.00141819*inputs[546] +0.00583168*inputs[547] +0.0202847*inputs[548] +0.00602022*inputs[549] +0.0027046*inputs[550] +0.0137369*inputs[551] -0.00381351*inputs[552] -0.0152014*inputs[553] +0.00261679*inputs[554] -0.000596465*inputs[555] +0.00761987*inputs[556] +0.00406753*inputs[557] +0.0130149*inputs[558] -0.0408029*inputs[559] +0.00210188*inputs[560] +0.00592165*inputs[561] +0.00965571*inputs[562] +0.0363419*inputs[563] +0.0476169*inputs[564] -0.000596465*inputs[565] -0.00661031*inputs[566] -0.0109676*inputs[567] -0.000596465*inputs[568] +0.0152495*inputs[569] +0.00939037*inputs[570] -0.000596465*inputs[571] -0.0122451*inputs[572] -0.0340908*inputs[573] -0.000596464*inputs[574] +0.0265953*inputs[575] -0.000596464*inputs[576] -0.0398394*inputs[577] +0.013721*inputs[578] -0.0250662*inputs[579] -0.00495376*inputs[580] -0.00150342*inputs[581] -0.000596465*inputs[582] -0.00850785*inputs[583] -0.00583515*inputs[584] -0.0205472*inputs[585] +0.0246307*inputs[586] -0.000596465*inputs[587] +0.0293423*inputs[588] -0.000596465*inputs[589] -0.00861525*inputs[590] -0.000421567*inputs[591] -0.000596464*inputs[592] -0.068496*inputs[593] -0.000934934*inputs[594] +0.0277302*inputs[595] -0.00454751*inputs[596] -0.0318027*inputs[597] -0.00509627*inputs[598] -0.00921329*inputs[599] +0.0198571*inputs[600] +0.0360884*inputs[601] -0.0120557*inputs[602] -0.0116546*inputs[603] -0.0183735*inputs[604] -0.000596465*inputs[605] +0.00479108*inputs[606] -0.00509627*inputs[607] +0.00839834*inputs[608] +0.0180955*inputs[609] -0.000596465*inputs[610] +0.0118572*inputs[611] +0.0320726*inputs[612] -0.00822287*inputs[613] -0.000421563*inputs[614] -0.00704951*inputs[615] +0.00189614*inputs[616] -0.0378501*inputs[617] -0.0141529*inputs[618] -0.00151508*inputs[619] +0.0222208*inputs[620] -0.00432836*inputs[621] -0.013085*inputs[622] +0.0222208*inputs[623] -0.000894275*inputs[624] +0.0125823*inputs[625] -0.0197077*inputs[626] -0.000596465*inputs[627] -0.000596464*inputs[628] -0.000596465*inputs[629] -0.0177619*inputs[630] -0.0174069*inputs[631] +0.00694876*inputs[632] -0.0358865*inputs[633] -0.00971476*inputs[634] -0.00557873*inputs[635] -0.000596465*inputs[636] +0.0108603*inputs[637] -0.0224744*inputs[638] -0.000596465*inputs[639] -0.0368168*inputs[640] +0.0151682*inputs[641] +0.00339774*inputs[642] +0.0259376*inputs[643] -0.0367261*inputs[644] -0.0177619*inputs[645] -0.0570269*inputs[646] -0.0120742*inputs[647] -0.000421569*inputs[648] +0.00893117*inputs[649] +0.00272093*inputs[650] +0.0288173*inputs[651] -0.00829336*inputs[652] +0.00893118*inputs[653] -0.00829339*inputs[654] -0.00259635*inputs[655] +0.00893118*inputs[656] -0.000421569*inputs[657] -0.000421569*inputs[658] +0.00272093*inputs[659] -0.000421569*inputs[660] -0.000421568*inputs[661] -0.00409046*inputs[662] -0.000421569*inputs[663] +0.00627279*inputs[664] +0.00325248*inputs[665] -0.0384217*inputs[666] -0.0234254*inputs[667] -0.000421569*inputs[668] +0.0244053*inputs[669] -0.00442924*inputs[670] -0.0244462*inputs[671] -0.000421569*inputs[672] -0.00409046*inputs[673] +0.00900415*inputs[674] +0.0138685*inputs[675] +0.00325248*inputs[676] -0.000421569*inputs[677] -0.000421566*inputs[678] -0.000421563*inputs[679] +0.00325248*inputs[680] -0.00259635*inputs[681] -0.000421569*inputs[682] +0.000385056*inputs[683] -0.000421569*inputs[684] -0.0323736*inputs[685] +0.0121758*inputs[686] -0.000421569*inputs[687] +0.0121759*inputs[688] +0.0121759*inputs[689] -0.000421569*inputs[690] -0.0221738*inputs[691] -0.000421569*inputs[692] -0.000421569*inputs[693] -0.00442923*inputs[694] -0.0049689*inputs[695] -0.00496889*inputs[696] -0.000421562*inputs[697] +0.0312304*inputs[698] +0.00900415*inputs[699] -0.0049689*inputs[700] -0.0049689*inputs[701] +0.0150434*inputs[702] -0.000421568*inputs[703] -0.000421569*inputs[704] -0.000421569*inputs[705] -0.000421568*inputs[706] -0.000421564*inputs[707] +0.0240649*inputs[708] -0.000421567*inputs[709] -0.000421564*inputs[710] -0.000421569*inputs[711] -0.000421568*inputs[712] +0.0150434*inputs[713] +0.0150434*inputs[714] +0.0150434*inputs[715] +0.00779636*inputs[716] -0.000421563*inputs[717] +0.0208774*inputs[718] +0.0208774*inputs[719] -0.000421566*inputs[720] -0.000421567*inputs[721] -0.000421559*inputs[722] -0.000421569*inputs[723] -0.000421569*inputs[724] -0.0043411*inputs[725] +0.00616447*inputs[726] +0.020803*inputs[727] -0.0149319*inputs[728] -0.0149319*inputs[729] -0.0149319*inputs[730] +0.020803*inputs[731] +0.0109947*inputs[732] -0.000421569*inputs[733] -0.000421565*inputs[734] -0.000421569*inputs[735] +0.00460719*inputs[736] +0.00460718*inputs[737] +0.00460718*inputs[738] +0.00460718*inputs[739] -0.00714915*inputs[740] +0.020803*inputs[741] -0.02301*inputs[742] -0.000421569*inputs[743] -0.000421569*inputs[744] -0.000421567*inputs[745] -0.00476562*inputs[746] -0.000421569*inputs[747] +0.0107111*inputs[748] -0.000421569*inputs[749] -0.000421568*inputs[750] +0.0234851*inputs[751] +0.0234851*inputs[752] -0.000421568*inputs[753] -0.000421569*inputs[754] -0.000421569*inputs[755] +0.0115702*inputs[756] -0.000421569*inputs[757] +0.0376014*inputs[758] -0.00476564*inputs[759] -0.0047656*inputs[760] -0.0136107*inputs[761] -0.0136107*inputs[762] -0.0136107*inputs[763] +0.0109947*inputs[764] -0.000421566*inputs[765] -0.000421566*inputs[766] -0.000421568*inputs[767] -0.000421569*inputs[768] -0.000421569*inputs[769] -0.000916001*inputs[770] -0.000421564*inputs[771] +0.00909814*inputs[772] -0.0210658*inputs[773] +0.00909814*inputs[774] +0.00928974*inputs[775] -0.00502791*inputs[776] +0.00909814*inputs[777] -0.00502791*inputs[778] -0.00502791*inputs[779] +0.000385056*inputs[780] +0.00795204*inputs[781] -0.0348484*inputs[782] -0.000916002*inputs[783] -0.000421566*inputs[784] +0.0265914*inputs[785] -0.00483046*inputs[786] -0.00483041*inputs[787] -0.00483044*inputs[788] -0.00483042*inputs[789] -0.0160948*inputs[790] +0.00773716*inputs[791] -0.000421569*inputs[792] -0.000421569*inputs[793] -0.0188879*inputs[794] -0.000421569*inputs[795] -0.0121627*inputs[796] -0.0121627*inputs[797] -0.000421567*inputs[798] -0.000421569*inputs[799] -0.000421569*inputs[800] -0.0141865*inputs[801] -0.0141865*inputs[802] -0.000421562*inputs[803] -0.00442921*inputs[804] -0.000421567*inputs[805] -0.0191421*inputs[806] +0.012873*inputs[807] +0.0333271*inputs[808] -0.000421569*inputs[809] +0.012873*inputs[810] -0.00746401*inputs[811] -0.00746401*inputs[812] -0.00746401*inputs[813] -0.00291374*inputs[814] -0.00291374*inputs[815] +0.00909814*inputs[816] -0.000421569*inputs[817] +0.00959415*inputs[818] -0.000421565*inputs[819] +0.00898304*inputs[820] +0.00898304*inputs[821] +0.0285953*inputs[822] -0.000421567*inputs[823] -0.000421568*inputs[824] +0.0328773*inputs[825] -0.0257271*inputs[826] -0.0257271*inputs[827] -0.000421569*inputs[828] -0.000421569*inputs[829] -0.000421568*inputs[830] -0.000421569*inputs[831] -0.000421569*inputs[832] +0.0140451*inputs[833] +0.0140451*inputs[834] +0.0140451*inputs[835] -0.000421569*inputs[836] -0.000421569*inputs[837] -0.000421569*inputs[838] -0.00192518*inputs[839] -0.0019252*inputs[840] -0.00192519*inputs[841] +0.0133185*inputs[842] -0.000421563*inputs[843] -0.000421569*inputs[844] +0.0294066*inputs[845] -0.0262668*inputs[846] -0.00986155*inputs[847] -0.000421561*inputs[848] +0.0116246*inputs[849] -0.0394874*inputs[850] +0.00719377*inputs[851] +0.00719377*inputs[852] +0.00719377*inputs[853] -0.000421569*inputs[854] +0.0195958*inputs[855] -0.000421569*inputs[856] -0.000421567*inputs[857] -0.000421569*inputs[858] -0.00367808*inputs[859] -0.00367808*inputs[860] -0.000421559*inputs[861] -0.0115078*inputs[862] -0.00576061*inputs[863] -0.0115078*inputs[864] -0.000421565*inputs[865] -0.0280121*inputs[866] -0.00328161*inputs[867] -0.00328161*inputs[868] +0.00781041*inputs[869] -0.000421566*inputs[870] -0.000421563*inputs[871] -0.000421561*inputs[872] -0.0392148*inputs[873] -0.000421562*inputs[874] -0.000421569*inputs[875] +0.0132485*inputs[876] +0.0132485*inputs[877] -0.020101*inputs[878] -0.0157061*inputs[879] -0.022743*inputs[880] -0.022743*inputs[881] -0.000421569*inputs[882] -0.029591*inputs[883] -0.029591*inputs[884] -0.0214953*inputs[885] -0.0231387*inputs[886] -0.000421569*inputs[887] -0.000421569*inputs[888] -0.000421569*inputs[889] -0.000421569*inputs[890] +0.0175041*inputs[891] -0.000421569*inputs[892] -0.000421567*inputs[893] +0.0158875*inputs[894] +0.00521886*inputs[895] -0.000421569*inputs[896] +0.00521886*inputs[897] -0.000421569*inputs[898] -0.00042156*inputs[899] -0.000421567*inputs[900] +0.0377212*inputs[901] +0.0376014*inputs[902] -0.000421568*inputs[903] +0.0116885*inputs[904] +0.00572173*inputs[905] -0.000421569*inputs[906] -0.000421568*inputs[907] +0.0238893*inputs[908] +0.0238893*inputs[909] -0.000421569*inputs[910] -0.000421569*inputs[911] -0.000421569*inputs[912] -0.000421568*inputs[913] -0.000421561*inputs[914] -0.000421569*inputs[915] +0.0177022*inputs[916] +0.0177022*inputs[917] +0.0177022*inputs[918] -0.000421569*inputs[919] -0.000421566*inputs[920] -0.000421568*inputs[921] -0.0179231*inputs[922] -0.0179231*inputs[923] +0.00522429*inputs[924] +0.00522429*inputs[925] +0.00522428*inputs[926] -0.000421569*inputs[927] -0.000421569*inputs[928] -0.000421568*inputs[929] +0.00866468*inputs[930] +0.00866468*inputs[931] +0.00866468*inputs[932] +0.00379624*inputs[933] +0.00379624*inputs[934] -0.000421569*inputs[935] -0.0206243*inputs[936] -0.0206243*inputs[937] -0.000421567*inputs[938] -0.000421563*inputs[939] +0.00795204*inputs[940] +0.0241509*inputs[941] +0.015883*inputs[942] +0.0309972*inputs[943] +0.00879186*inputs[944] +0.00879186*inputs[945] -0.000421568*inputs[946] -0.000421569*inputs[947] -0.00226737*inputs[948] -0.00226737*inputs[949] +0.0290941*inputs[950] -0.0271072*inputs[951] -0.000421568*inputs[952] -0.00042156*inputs[953] -0.000421569*inputs[954] -0.000421569*inputs[955] -0.000421567*inputs[956] -0.000421569*inputs[957] -0.000421569*inputs[958] -0.000421569*inputs[959] -0.000421569*inputs[960] -0.000421567*inputs[961] -0.000421569*inputs[962] -0.000421564*inputs[963] +0.0222679*inputs[964] +0.0187713*inputs[965] +0.0279232*inputs[966] +0.0185327*inputs[967] -0.000421565*inputs[968] -0.000421569*inputs[969] +0.00775037*inputs[970] +0.00775037*inputs[971] -0.0272877*inputs[972] -0.0272877*inputs[973] -0.000421567*inputs[974] +0.0222679*inputs[975] +0.00350116*inputs[976] -0.0270213*inputs[977] +0.00350116*inputs[978] -0.0267622*inputs[979] -0.0272192*inputs[980] -0.0110282*inputs[981] -0.000421562*inputs[982] -0.00921234*inputs[983] -0.00921234*inputs[984] +0.00900414*inputs[985] -0.00336964*inputs[986] -0.0116043*inputs[987] -0.0255523*inputs[988] -0.000421569*inputs[989] -0.000421568*inputs[990] -0.0082529*inputs[991] -0.0082529*inputs[992] -0.0082529*inputs[993] +0.0173588*inputs[994] -0.000421569*inputs[995] -0.000421566*inputs[996] -0.0138976*inputs[997] -0.0174667*inputs[998] -0.00688428*inputs[999] -0.000421566*inputs[1000] +0.024436*inputs[1001] -0.000421569*inputs[1002] -0.0106693*inputs[1003] -0.0106693*inputs[1004] -0.000421568*inputs[1005] -0.000421568*inputs[1006] -0.000421568*inputs[1007] -0.000421569*inputs[1008] -0.000421569*inputs[1009] -0.0289967*inputs[1010] +0.0389883*inputs[1011] -0.0270213*inputs[1012] -0.000421567*inputs[1013] +0.012191*inputs[1014] -0.000421568*inputs[1015] -0.000421569*inputs[1016] +0.00816105*inputs[1017] +0.00816105*inputs[1018] -0.000421562*inputs[1019] -0.0293602*inputs[1020] -0.000421566*inputs[1021] +0.00533647*inputs[1022] -0.000421562*inputs[1023] +0.012191*inputs[1024] -0.000421569*inputs[1025] +0.012191*inputs[1026] -0.0160034*inputs[1027] -0.000421565*inputs[1028] +0.00767494*inputs[1029] -0.0294197*inputs[1030] -0.0165075*inputs[1031] -0.0165075*inputs[1032] +0.00767494*inputs[1033] -0.000421569*inputs[1034] -0.000421569*inputs[1035] +0.00767495*inputs[1036] +0.00780376*inputs[1037] -0.0108842*inputs[1038] -0.000421569*inputs[1039] -0.0222293*inputs[1040] -0.000421569*inputs[1041] -0.00703357*inputs[1042] -0.00703353*inputs[1043] +0.0112682*inputs[1044] +0.0112682*inputs[1045] +0.0112682*inputs[1046] +0.00780376*inputs[1047] +0.00780376*inputs[1048] -0.00883908*inputs[1049] +0.0194837*inputs[1050] -0.0148372*inputs[1051] -0.0148372*inputs[1052] -0.000421569*inputs[1053] -0.0326437*inputs[1054] -0.000421566*inputs[1055] -0.000421561*inputs[1056] -0.0423006*inputs[1057] -0.000421564*inputs[1058] -0.000421566*inputs[1059] -0.000421569*inputs[1060] -0.0112723*inputs[1061] +0.00642179*inputs[1062] -0.00600634*inputs[1063] -0.00600635*inputs[1064] -0.000421568*inputs[1065] -0.000421568*inputs[1066] -0.0183102*inputs[1067] -0.0183102*inputs[1068] -0.00714913*inputs[1069] -0.00175911*inputs[1070] +0.0274096*inputs[1071] -0.0112723*inputs[1072] -0.000421564*inputs[1073] -0.000421569*inputs[1074] -0.000421566*inputs[1075] -0.000421569*inputs[1076] -0.00379682*inputs[1077] -0.00379682*inputs[1078] -0.00379682*inputs[1079] +0.00317162*inputs[1080] +0.00317162*inputs[1081] -0.000421568*inputs[1082] -0.000421564*inputs[1083] -0.00813306*inputs[1084] -0.000421564*inputs[1085] -0.000421568*inputs[1086] -0.000421569*inputs[1087] -0.000421564*inputs[1088] -0.000421569*inputs[1089] +0.00767494*inputs[1090] +0.0309053*inputs[1091] -0.000421569*inputs[1092] +0.0141316*inputs[1093] -0.000421568*inputs[1094] -0.024685*inputs[1095] +0.0252452*inputs[1096] -0.000421569*inputs[1097] +0.0170125*inputs[1098] -0.000421569*inputs[1099] -0.0160523*inputs[1100] -0.0356234*inputs[1101] +0.039244*inputs[1102] +0.00722494*inputs[1103] +0.00807751*inputs[1104] -0.0149053*inputs[1105] -0.000421569*inputs[1106] +0.00722495*inputs[1107] -0.000421569*inputs[1108] +0.0167395*inputs[1109] -0.000421567*inputs[1110] -0.0112016*inputs[1111] -0.000421562*inputs[1112] +0.0192623*inputs[1113] -0.00733176*inputs[1114] -0.016835*inputs[1115] +0.0167395*inputs[1116] +0.00730501*inputs[1117] +0.0262574*inputs[1118] -0.000421567*inputs[1119] -0.000421566*inputs[1120] -0.00580092*inputs[1121] -0.0111044*inputs[1122] +0.0324435*inputs[1123] -0.000421561*inputs[1124] +0.00229312*inputs[1125] -0.00733176*inputs[1126] +0.0079661*inputs[1127] -0.000421568*inputs[1128] +0.00730501*inputs[1129] -0.000421567*inputs[1130] -0.000421569*inputs[1131] -0.000421563*inputs[1132] -0.000421569*inputs[1133] -0.00042156*inputs[1134] -0.00590476*inputs[1135] -0.000421569*inputs[1136] +0.0157727*inputs[1137] -0.0345143*inputs[1138] -0.00733176*inputs[1139] -0.000421569*inputs[1140] +0.00522284*inputs[1141] +0.00522284*inputs[1142] -0.000421569*inputs[1143] -0.00733176*inputs[1144] -0.000421569*inputs[1145] -0.000421561*inputs[1146] -0.000421568*inputs[1147] -0.000421566*inputs[1148] -0.000421569*inputs[1149] +0.0207707*inputs[1150] -0.000421568*inputs[1151] -0.000421565*inputs[1152] -0.000421569*inputs[1153] -0.00580092*inputs[1154] -0.00580093*inputs[1155] -0.000421569*inputs[1156] -0.0149676*inputs[1157] +0.0110956*inputs[1158] +0.0110956*inputs[1159] -0.0204316*inputs[1160] +0.0102436*inputs[1161] +0.0249852*inputs[1162] -0.000421569*inputs[1163] -0.000421564*inputs[1164] +0.0102436*inputs[1165] -0.0291063*inputs[1166] -0.000421569*inputs[1167] -0.000421569*inputs[1168] +0.0110956*inputs[1169] -0.00788988*inputs[1170] -0.000421569*inputs[1171] -0.011001*inputs[1172] -0.000421569*inputs[1173] -0.00788991*inputs[1174] -0.000421569*inputs[1175] -0.000421569*inputs[1176] -0.000421563*inputs[1177] -0.0149676*inputs[1178] -0.000421569*inputs[1179] -0.000421569*inputs[1180] +0.011664*inputs[1181] -0.000421569*inputs[1182] -0.000421567*inputs[1183] -0.000421569*inputs[1184] -0.000421569*inputs[1185] -0.000421562*inputs[1186] -0.00733176*inputs[1187] -0.00352513*inputs[1188] -0.00352513*inputs[1189] -0.00921329*inputs[1190] +0.011664*inputs[1191] +0.011664*inputs[1192] -0.0144315*inputs[1193] +0.011664*inputs[1194] -0.000421569*inputs[1195] -0.000421565*inputs[1196] -0.00921329*inputs[1197] -0.0168244*inputs[1198] -0.0168244*inputs[1199] +0.0345463*inputs[1200] -0.000421568*inputs[1201] -0.000421565*inputs[1202] -0.000421568*inputs[1203] -0.000421569*inputs[1204] +0.00770361*inputs[1205] -0.0117562*inputs[1206] -0.000421566*inputs[1207] -0.000421563*inputs[1208] -0.000421569*inputs[1209] -0.000421569*inputs[1210] -0.000421568*inputs[1211] +0.00226602*inputs[1212] -0.0250953*inputs[1213] +0.00226602*inputs[1214] -0.000421569*inputs[1215] -0.000421567*inputs[1216] -0.000421569*inputs[1217] +0.0077036*inputs[1218] -0.0268425*inputs[1219] -0.000421569*inputs[1220] +0.00797265*inputs[1221] +0.00797266*inputs[1222] +0.00797265*inputs[1223] +0.00797265*inputs[1224] -0.0100176*inputs[1225] -0.000421569*inputs[1226] -0.000421569*inputs[1227] +0.00330594*inputs[1228] -0.000421569*inputs[1229] -0.00844306*inputs[1230] -0.00432102*inputs[1231] -0.00432102*inputs[1232] +0.00412041*inputs[1233] +0.00412041*inputs[1234] +0.00308497*inputs[1235] -0.000381482*inputs[1236] -0.000421569*inputs[1237] -0.000381483*inputs[1238] -0.000381483*inputs[1239] +0.00308497*inputs[1240] +0.00330594*inputs[1241] +0.0118665*inputs[1242] -0.000421568*inputs[1243] +0.0186467*inputs[1244] +0.0186467*inputs[1245] -0.000421569*inputs[1246] -0.000421567*inputs[1247] +0.0179905*inputs[1248] +0.0179905*inputs[1249] -0.000421569*inputs[1250] -0.000421569*inputs[1251] -0.000421569*inputs[1252] -0.00185393*inputs[1253] -0.000421569*inputs[1254] -0.000421567*inputs[1255] -0.000421569*inputs[1256] -0.000421567*inputs[1257] -0.00658057*inputs[1258] -0.000421567*inputs[1259] -0.00658057*inputs[1260] -0.0305739*inputs[1261] -0.000421569*inputs[1262] -0.000421569*inputs[1263] -0.00185393*inputs[1264] -0.000421569*inputs[1265] -0.00185392*inputs[1266] -0.0103039*inputs[1267] -0.000421569*inputs[1268] +0.0171818*inputs[1269] -0.000421569*inputs[1270] -0.000421567*inputs[1271] -0.0213768*inputs[1272] -0.000421562*inputs[1273] -0.000421567*inputs[1274] -0.000421563*inputs[1275] -0.000421567*inputs[1276] -0.00499833*inputs[1277] -0.000421568*inputs[1278] +0.0302013*inputs[1279] -0.000421563*inputs[1280] +0.0134523*inputs[1281] -0.000421569*inputs[1282] -0.000421569*inputs[1283] +0.0167768*inputs[1284] -0.000421568*inputs[1285] -0.00499833*inputs[1286] -0.00499833*inputs[1287] -0.00499833*inputs[1288] -0.000421569*inputs[1289] -0.0377685*inputs[1290] -0.0286221*inputs[1291] -0.000421569*inputs[1292] -0.0214728*inputs[1293] -0.000421567*inputs[1294] +0.0390969*inputs[1295] +0.000628652*inputs[1296] +0.000628652*inputs[1297] -0.000421569*inputs[1298] -0.00782651*inputs[1299] -0.000421569*inputs[1300] -0.00452004*inputs[1301] +0.0122391*inputs[1302] -0.00454842*inputs[1303] -0.000421565*inputs[1304] -0.000421569*inputs[1305] -0.0317988*inputs[1306] -0.000421569*inputs[1307] -0.000421569*inputs[1308] +0.00185782*inputs[1309] -0.000421569*inputs[1310] -0.00452004*inputs[1311] -0.00452004*inputs[1312] -0.00454844*inputs[1313] -0.00452005*inputs[1314] -0.00452004*inputs[1315] -0.000421569*inputs[1316] -0.000421569*inputs[1317] +0.00339256*inputs[1318] -0.000421569*inputs[1319] -0.000421569*inputs[1320] -0.000421563*inputs[1321] +0.0193442*inputs[1322] +0.0106883*inputs[1323] +0.0106883*inputs[1324] +0.0103928*inputs[1325] +0.00982932*inputs[1326] +0.00982932*inputs[1327] +0.00982932*inputs[1328] +0.00982932*inputs[1329] +0.0163418*inputs[1330] +0.0237217*inputs[1331] +0.00337426*inputs[1332] +0.00337427*inputs[1333] -0.000421569*inputs[1334] -0.000421569*inputs[1335] -0.000421569*inputs[1336] -0.0170874*inputs[1337] -0.000421569*inputs[1338] -0.00813306*inputs[1339] -0.0291683*inputs[1340] -0.00042156*inputs[1341] -0.00457919*inputs[1342] -0.00457917*inputs[1343] -0.000421563*inputs[1344] -0.000421567*inputs[1345] -0.000421568*inputs[1346] -0.000421569*inputs[1347] -0.00454842*inputs[1348] +0.00748227*inputs[1349] -0.0274355*inputs[1350] -0.0112483*inputs[1351] -0.0112483*inputs[1352] +0.00339256*inputs[1353] +0.0124444*inputs[1354] -0.00840764*inputs[1355] -0.00840762*inputs[1356] -0.00840765*inputs[1357] +0.00748228*inputs[1358] +0.00748227*inputs[1359] +0.0137894*inputs[1360] +0.00339256*inputs[1361] -0.000421568*inputs[1362] -0.00432103*inputs[1363] -0.00714914*inputs[1364] +0.0111923*inputs[1365] -0.00432101*inputs[1366] -0.000421567*inputs[1367] +0.0111923*inputs[1368] +0.0111923*inputs[1369] -0.000421564*inputs[1370] -0.00432103*inputs[1371] -0.000421562*inputs[1372] -0.00852673*inputs[1373] -0.000421569*inputs[1374] -0.0167964*inputs[1375] -0.0117217*inputs[1376] -0.00539507*inputs[1377] -0.00539507*inputs[1378] -0.00539507*inputs[1379] -0.000421569*inputs[1380] -0.00892227*inputs[1381] -0.000421567*inputs[1382] -0.0377438*inputs[1383] -0.000421568*inputs[1384] -0.000421563*inputs[1385] -0.00852673*inputs[1386] -0.00843003*inputs[1387] -0.00843003*inputs[1388] -0.00843003*inputs[1389] -0.000421568*inputs[1390] -0.000421569*inputs[1391] -0.000421549*inputs[1392] +0.00339257*inputs[1393] -0.000421553*inputs[1394] +0.0203695*inputs[1395] 
		combinations[2] = -0.0134108 -0.0566944*inputs[0] -0.0581455*inputs[1] -0.236143*inputs[2] -0.163095*inputs[3] -0.0906439*inputs[4] -0.0791161*inputs[5] -0.0461878*inputs[6] -0.00273575*inputs[7] -0.0298067*inputs[8] +0.000862318*inputs[9] -0.00255934*inputs[10] -0.108016*inputs[11] -0.0266535*inputs[12] -0.112259*inputs[13] -0.0459905*inputs[14] -0.126158*inputs[15] +0.00339169*inputs[16] -0.0109767*inputs[17] -0.105807*inputs[18] -0.00175611*inputs[19] -0.0412272*inputs[20] +0.0866152*inputs[21] -0.0288611*inputs[22] -0.0330539*inputs[23] -0.12703*inputs[24] +0.00876871*inputs[25] +0.0672929*inputs[26] +0.00977787*inputs[27] +0.00996175*inputs[28] +0.0422475*inputs[29] +0.00913587*inputs[30] +0.074478*inputs[31] +0.0368416*inputs[32] -0.0979287*inputs[33] +0.054838*inputs[34] -0.110406*inputs[35] -0.0706789*inputs[36] -0.0409324*inputs[37] -0.0133541*inputs[38] +0.114837*inputs[39] +0.0589462*inputs[40] -0.13193*inputs[41] -0.0607172*inputs[42] -0.043042*inputs[43] +0.0120326*inputs[44] -0.0156508*inputs[45] +0.0359839*inputs[46] -0.0753423*inputs[47] +0.0415915*inputs[48] +0.0045436*inputs[49] -0.105053*inputs[50] -0.0107682*inputs[51] +0.0180355*inputs[52] +0.0167753*inputs[53] -0.00651766*inputs[54] +0.0628273*inputs[55] +0.0264319*inputs[56] -0.031287*inputs[57] +0.110777*inputs[58] +0.00509895*inputs[59] -0.0239936*inputs[60] +0.118617*inputs[61] +0.0383953*inputs[62] +0.0243455*inputs[63] +0.0423277*inputs[64] +0.0226038*inputs[65] -0.0728554*inputs[66] +0.0637808*inputs[67] -0.0223177*inputs[68] +0.0111142*inputs[69] +0.011798*inputs[70] -0.023191*inputs[71] +0.00253394*inputs[72] -0.0277504*inputs[73] +0.0880994*inputs[74] -0.00712892*inputs[75] -0.0265667*inputs[76] +0.0369027*inputs[77] +0.184444*inputs[78] -0.0244973*inputs[79] +0.00643897*inputs[80] -0.0136149*inputs[81] -0.0784764*inputs[82] +0.000586032*inputs[83] -0.0146132*inputs[84] +0.0274737*inputs[85] -0.0431154*inputs[86] +0.0440152*inputs[87] +0.0261228*inputs[88] -0.0129514*inputs[89] +0.0122485*inputs[90] -0.0308515*inputs[91] +0.0549781*inputs[92] +0.06772*inputs[93] -0.0579072*inputs[94] +0.0660868*inputs[95] +0.0141334*inputs[96] +0.0405361*inputs[97] +0.0100989*inputs[98] -0.00781787*inputs[99] +0.00597424*inputs[100] +0.00591901*inputs[101] -0.00900981*inputs[102] +0.0646859*inputs[103] +0.0233468*inputs[104] -1.03146e-05*inputs[105] -0.0169181*inputs[106] -0.0130416*inputs[107] -0.0160164*inputs[108] +0.00450819*inputs[109] -0.0500401*inputs[110] -0.0614245*inputs[111] -0.025296*inputs[112] +0.00871147*inputs[113] -0.075988*inputs[114] +0.0411654*inputs[115] +0.0534132*inputs[116] -0.00192333*inputs[117] -0.0467079*inputs[118] +0.0178517*inputs[119] +0.0167282*inputs[120] +0.00261969*inputs[121] +0.0581687*inputs[122] +0.00203103*inputs[123] +0.00895966*inputs[124] +0.01714*inputs[125] +0.00477957*inputs[126] +0.0114435*inputs[127] +0.0667905*inputs[128] -0.0251943*inputs[129] +0.00495727*inputs[130] -0.0384024*inputs[131] -0.0258361*inputs[132] -0.0913257*inputs[133] +0.0181578*inputs[134] +0.0237034*inputs[135] +0.00942601*inputs[136] +0.0475845*inputs[137] -0.00725866*inputs[138] -0.00935839*inputs[139] -0.00632525*inputs[140] -0.0286279*inputs[141] -0.0208153*inputs[142] +0.108155*inputs[143] -0.0105324*inputs[144] +0.0348955*inputs[145] +0.0085113*inputs[146] -0.00840836*inputs[147] +0.00738374*inputs[148] +0.00131136*inputs[149] +0.0162771*inputs[150] -0.0637271*inputs[151] +0.0548707*inputs[152] +0.0387798*inputs[153] -0.0121681*inputs[154] +0.0016256*inputs[155] +0.0235546*inputs[156] +0.00866614*inputs[157] +0.0609045*inputs[158] -0.0154777*inputs[159] -0.0352229*inputs[160] +0.0111287*inputs[161] +0.0180833*inputs[162] +0.00521908*inputs[163] +0.0353779*inputs[164] +0.0054312*inputs[165] +0.0102016*inputs[166] +0.0368223*inputs[167] +0.0250825*inputs[168] -0.0546547*inputs[169] +0.0117093*inputs[170] +0.0201036*inputs[171] +0.0309498*inputs[172] +0.0200871*inputs[173] +0.00409946*inputs[174] -0.00618786*inputs[175] +0.028586*inputs[176] -0.0120318*inputs[177] +0.0333414*inputs[178] -0.00894319*inputs[179] +0.0362219*inputs[180] -0.00376135*inputs[181] +0.0518311*inputs[182] +0.00654849*inputs[183] +0.0852117*inputs[184] -0.018907*inputs[185] +0.038339*inputs[186] -0.0172895*inputs[187] -0.00652172*inputs[188] +0.0238323*inputs[189] -0.0746553*inputs[190] -0.0458269*inputs[191] -0.0110542*inputs[192] -0.0198508*inputs[193] +0.00421766*inputs[194] +0.00355187*inputs[195] +0.00410283*inputs[196] +0.00740962*inputs[197] -0.0249438*inputs[198] -0.0368798*inputs[199] -0.0067506*inputs[200] -0.0552391*inputs[201] -0.0230202*inputs[202] +0.0154528*inputs[203] +0.00263729*inputs[204] -0.00699415*inputs[205] -0.0335072*inputs[206] +0.00220612*inputs[207] +0.0230806*inputs[208] -0.0530855*inputs[209] +0.058776*inputs[210] -0.00829049*inputs[211] +0.000950179*inputs[212] -0.00884504*inputs[213] -0.0228915*inputs[214] +0.00313729*inputs[215] +0.0146462*inputs[216] -0.00622575*inputs[217] -0.0392841*inputs[218] +0.00300148*inputs[219] +0.0211808*inputs[220] -0.0183576*inputs[221] +0.00746275*inputs[222] +0.0476394*inputs[223] -0.00936633*inputs[224] +0.0473119*inputs[225] -0.0166332*inputs[226] +0.0102221*inputs[227] +0.0274868*inputs[228] +0.0624224*inputs[229] -0.0281762*inputs[230] +0.0138046*inputs[231] -0.00106622*inputs[232] +0.00587239*inputs[233] +0.00660141*inputs[234] +0.00460971*inputs[235] +0.0234593*inputs[236] -0.00444684*inputs[237] -0.0170779*inputs[238] +0.0164556*inputs[239] -0.0304296*inputs[240] -0.0209354*inputs[241] +0.00725082*inputs[242] -0.0577602*inputs[243] -0.0207799*inputs[244] +0.0484468*inputs[245] -0.010183*inputs[246] -0.0207981*inputs[247] +0.0239417*inputs[248] +0.0569137*inputs[249] +0.0141838*inputs[250] +0.0330195*inputs[251] -0.0146081*inputs[252] -0.0210376*inputs[253] -0.00764096*inputs[254] -0.0341067*inputs[255] -0.0429467*inputs[256] -0.0776566*inputs[257] -0.0166122*inputs[258] -0.0372937*inputs[259] +0.0210934*inputs[260] +0.0140053*inputs[261] +0.0523624*inputs[262] -0.0323997*inputs[263] -0.00914327*inputs[264] +0.0193756*inputs[265] +0.0182006*inputs[266] +0.00599465*inputs[267] -0.014265*inputs[268] +0.0146518*inputs[269] -0.0372481*inputs[270] -0.0647833*inputs[271] +0.00476613*inputs[272] -0.000626171*inputs[273] +0.0086634*inputs[274] -0.0310446*inputs[275] +0.0371292*inputs[276] +0.0442366*inputs[277] +0.00327857*inputs[278] +0.0528506*inputs[279] -0.0832822*inputs[280] +0.00420216*inputs[281] -0.00600169*inputs[282] +0.00583025*inputs[283] -0.00327686*inputs[284] +0.0177841*inputs[285] +0.0143786*inputs[286] -0.0664246*inputs[287] -0.00172315*inputs[288] +0.00883739*inputs[289] +0.000336741*inputs[290] +0.00460282*inputs[291] +0.00513844*inputs[292] +0.00312583*inputs[293] +0.00487011*inputs[294] -0.0165266*inputs[295] -0.017544*inputs[296] +0.0161564*inputs[297] +0.0016828*inputs[298] +0.00970263*inputs[299] +0.00107252*inputs[300] +0.00858084*inputs[301] +0.00899681*inputs[302] -0.0167491*inputs[303] +0.00375973*inputs[304] -0.00629435*inputs[305] +0.0276615*inputs[306] +0.00725233*inputs[307] +0.00196007*inputs[308] +0.0135877*inputs[309] -0.000733501*inputs[310] +0.0170783*inputs[311] -0.0149235*inputs[312] +0.0179675*inputs[313] +0.0335561*inputs[314] -0.00336994*inputs[315] +0.0149564*inputs[316] +0.0242387*inputs[317] +0.0084023*inputs[318] +0.0321846*inputs[319] -0.0339981*inputs[320] -0.00721588*inputs[321] +0.000569204*inputs[322] -0.00696686*inputs[323] -0.0105179*inputs[324] +0.0261031*inputs[325] +0.000735252*inputs[326] +0.0105152*inputs[327] -0.0175447*inputs[328] +0.0136314*inputs[329] +0.0116404*inputs[330] +0.0144653*inputs[331] -0.0109071*inputs[332] +0.00053642*inputs[333] -0.0266624*inputs[334] +0.0181172*inputs[335] +0.0171205*inputs[336] -0.0284289*inputs[337] +0.00828514*inputs[338] +0.00218757*inputs[339] -0.0184074*inputs[340] +0.0325093*inputs[341] +0.0278949*inputs[342] +0.0165007*inputs[343] +0.0150817*inputs[344] +0.00574579*inputs[345] +0.0153027*inputs[346] +0.0544748*inputs[347] +0.0318492*inputs[348] +0.014557*inputs[349] -0.0195625*inputs[350] +0.000735252*inputs[351] +0.0101066*inputs[352] -0.0293341*inputs[353] +0.0015447*inputs[354] +0.0100343*inputs[355] -0.0127225*inputs[356] -0.0120952*inputs[357] -0.0134138*inputs[358] -0.0237296*inputs[359] +0.00838903*inputs[360] +0.00909278*inputs[361] +0.0108975*inputs[362] -0.0184072*inputs[363] +0.00336091*inputs[364] +0.0175607*inputs[365] -0.00720446*inputs[366] +0.00629142*inputs[367] +0.00882275*inputs[368] +0.0250622*inputs[369] -0.0443689*inputs[370] +0.00456171*inputs[371] +0.00238726*inputs[372] +0.066234*inputs[373] +0.0177118*inputs[374] +0.000735252*inputs[375] +0.0157479*inputs[376] -0.0382396*inputs[377] +0.0423103*inputs[378] +0.0294819*inputs[379] +0.0114788*inputs[380] +0.00512877*inputs[381] -0.00384959*inputs[382] +0.0175709*inputs[383] +0.0398575*inputs[384] -0.00940237*inputs[385] +0.011484*inputs[386] +0.00420769*inputs[387] -0.000506433*inputs[388] -0.0146002*inputs[389] -0.00236199*inputs[390] +0.000735252*inputs[391] -0.0129568*inputs[392] -0.023622*inputs[393] +0.000735252*inputs[394] -0.014854*inputs[395] -0.00318852*inputs[396] +0.0453782*inputs[397] +0.0325121*inputs[398] +0.0418878*inputs[399] +0.0159393*inputs[400] +0.000735252*inputs[401] +0.00702818*inputs[402] +0.0131976*inputs[403] -0.0035026*inputs[404] -0.0504058*inputs[405] +0.00226471*inputs[406] -0.0145367*inputs[407] -0.0291908*inputs[408] -0.0229742*inputs[409] +0.0149995*inputs[410] -0.00986099*inputs[411] +0.0146093*inputs[412] -0.00321471*inputs[413] +0.00966924*inputs[414] -0.0233491*inputs[415] +0.0402126*inputs[416] -0.0262416*inputs[417] +0.0251805*inputs[418] -0.00588641*inputs[419] +0.0118273*inputs[420] +0.0453194*inputs[421] +0.00621217*inputs[422] -0.00731489*inputs[423] +0.002866*inputs[424] +0.000735252*inputs[425] -0.00469473*inputs[426] +0.0157632*inputs[427] -0.00399876*inputs[428] +0.000134287*inputs[429] +0.000735253*inputs[430] +0.0227943*inputs[431] -0.00926008*inputs[432] +0.00311906*inputs[433] -0.0152081*inputs[434] +0.00225907*inputs[435] -0.0167017*inputs[436] -0.0125499*inputs[437] +0.00532991*inputs[438] -0.0185758*inputs[439] +0.0201653*inputs[440] +0.0299387*inputs[441] +0.00629879*inputs[442] +0.0186376*inputs[443] +0.000735251*inputs[444] +0.0184014*inputs[445] +0.00529163*inputs[446] +0.0233099*inputs[447] +0.000104019*inputs[448] +0.00060003*inputs[449] -0.0200405*inputs[450] -0.0355178*inputs[451] +0.0032083*inputs[452] +0.0328202*inputs[453] -0.00505078*inputs[454] +0.0272262*inputs[455] +0.035846*inputs[456] +0.013891*inputs[457] -0.0577013*inputs[458] +0.0504653*inputs[459] +0.0294748*inputs[460] -0.0042162*inputs[461] +0.00350198*inputs[462] +0.0209459*inputs[463] +0.0358466*inputs[464] +0.0550287*inputs[465] -0.000275637*inputs[466] +0.0105473*inputs[467] +0.013303*inputs[468] +0.00590284*inputs[469] +0.0162376*inputs[470] +0.0217268*inputs[471] -0.0107835*inputs[472] +0.00895255*inputs[473] -0.0182565*inputs[474] -0.0182566*inputs[475] +0.0134696*inputs[476] +0.0255493*inputs[477] -0.0102474*inputs[478] +0.010786*inputs[479] -0.00517669*inputs[480] -0.04021*inputs[481] +0.0249762*inputs[482] +0.0212534*inputs[483] +0.000600033*inputs[484] +0.00823951*inputs[485] +0.0247487*inputs[486] +0.0222257*inputs[487] +0.0106091*inputs[488] +0.0174778*inputs[489] -0.00817732*inputs[490] +0.0239589*inputs[491] +0.0116647*inputs[492] +0.0312499*inputs[493] +0.0108397*inputs[494] +0.00748263*inputs[495] +0.00748264*inputs[496] +0.0112401*inputs[497] -0.0695313*inputs[498] +0.00627876*inputs[499] +0.0212534*inputs[500] +0.00895256*inputs[501] +0.000600033*inputs[502] +0.000600032*inputs[503] +0.0271113*inputs[504] -0.0381897*inputs[505] -0.020052*inputs[506] -0.0145369*inputs[507] +0.000600033*inputs[508] +0.00385322*inputs[509] -0.00200953*inputs[510] +0.0520429*inputs[511] +0.0181493*inputs[512] -0.00604698*inputs[513] +0.000600033*inputs[514] -0.0290648*inputs[515] +0.000600031*inputs[516] +0.0125193*inputs[517] +0.013891*inputs[518] +0.000600033*inputs[519] -0.00200953*inputs[520] -0.00629287*inputs[521] +0.018205*inputs[522] -0.00236315*inputs[523] +0.00559432*inputs[524] +0.00559432*inputs[525] -0.00733042*inputs[526] +0.0145674*inputs[527] +0.000600033*inputs[528] -0.00296613*inputs[529] +0.00853859*inputs[530] -0.0207069*inputs[531] -0.0108096*inputs[532] -0.0109906*inputs[533] +0.017139*inputs[534] +0.0258731*inputs[535] +0.014469*inputs[536] +0.0258433*inputs[537] -0.0109906*inputs[538] -0.0116591*inputs[539] +0.000600032*inputs[540] +0.0119205*inputs[541] +0.0133079*inputs[542] -0.00217138*inputs[543] +0.000600033*inputs[544] +0.00853858*inputs[545] +0.00141707*inputs[546] -0.00585213*inputs[547] -0.0204058*inputs[548] -0.00603942*inputs[549] -0.00271697*inputs[550] -0.0138484*inputs[551] +0.00382002*inputs[552] +0.0152645*inputs[553] -0.00261802*inputs[554] +0.000600031*inputs[555] -0.00765839*inputs[556] -0.00405548*inputs[557] -0.0130588*inputs[558] +0.04102*inputs[559] -0.00209685*inputs[560] -0.00593696*inputs[561] -0.00972213*inputs[562] -0.0365588*inputs[563] -0.0479329*inputs[564] +0.000600028*inputs[565] +0.00661948*inputs[566] +0.0109841*inputs[567] +0.000600033*inputs[568] -0.0153075*inputs[569] -0.00940985*inputs[570] +0.000600033*inputs[571] +0.0122952*inputs[572] +0.0342393*inputs[573] +0.000600031*inputs[574] -0.0266793*inputs[575] +0.000600031*inputs[576] +0.0400248*inputs[577] -0.0137967*inputs[578] +0.0251741*inputs[579] +0.00496464*inputs[580] +0.00148959*inputs[581] +0.000600032*inputs[582] +0.00853858*inputs[583] +0.00583707*inputs[584] +0.0206502*inputs[585] -0.0248067*inputs[586] +0.000600031*inputs[587] -0.0295067*inputs[588] +0.000600033*inputs[589] +0.00864176*inputs[590] +0.000424081*inputs[591] +0.000600032*inputs[592] +0.068972*inputs[593] +0.000946323*inputs[594] -0.027887*inputs[595] +0.00457018*inputs[596] +0.0319672*inputs[597] +0.005091*inputs[598] +0.00923864*inputs[599] -0.0199461*inputs[600] -0.0363352*inputs[601] +0.0120959*inputs[602] +0.0117103*inputs[603] +0.0184856*inputs[604] +0.000600029*inputs[605] -0.00479775*inputs[606] +0.005091*inputs[607] -0.00840255*inputs[608] -0.0181358*inputs[609] +0.000600033*inputs[610] -0.0119155*inputs[611] -0.0322584*inputs[612] +0.00823951*inputs[613] +0.00042408*inputs[614] +0.00708277*inputs[615] -0.00191741*inputs[616] +0.038065*inputs[617] +0.0141844*inputs[618] +0.00150743*inputs[619] -0.0223496*inputs[620] +0.00433812*inputs[621] +0.0131079*inputs[622] -0.0223496*inputs[623] +0.000894217*inputs[624] -0.0126396*inputs[625] +0.019795*inputs[626] +0.000600033*inputs[627] +0.000600031*inputs[628] +0.000600032*inputs[629] +0.0178739*inputs[630] +0.0175205*inputs[631] -0.00696961*inputs[632] +0.0360478*inputs[633] +0.00974802*inputs[634] +0.00559432*inputs[635] +0.00060003*inputs[636] -0.0109073*inputs[637] +0.0226*inputs[638] +0.000600033*inputs[639] +0.0370176*inputs[640] -0.0152406*inputs[641] -0.00339841*inputs[642] -0.0261152*inputs[643] +0.0368687*inputs[644] +0.0178739*inputs[645] +0.0573578*inputs[646] +0.0121014*inputs[647] +0.000424078*inputs[648] -0.00896084*inputs[649] -0.00270334*inputs[650] -0.0289472*inputs[651] +0.00830809*inputs[652] -0.00896084*inputs[653] +0.0083081*inputs[654] +0.0025881*inputs[655] -0.00896084*inputs[656] +0.000424074*inputs[657] +0.000424077*inputs[658] -0.00270334*inputs[659] +0.000424077*inputs[660] +0.000424075*inputs[661] +0.00411084*inputs[662] +0.000424081*inputs[663] -0.00624627*inputs[664] -0.00325091*inputs[665] +0.0386897*inputs[666] +0.0235068*inputs[667] +0.00042408*inputs[668] -0.0245074*inputs[669] +0.00442392*inputs[670] +0.0245257*inputs[671] +0.000424081*inputs[672] +0.00411084*inputs[673] -0.00901183*inputs[674] -0.0139142*inputs[675] -0.00325091*inputs[676] +0.000424074*inputs[677] +0.000424081*inputs[678] +0.000424074*inputs[679] -0.00325091*inputs[680] +0.0025881*inputs[681] +0.000424078*inputs[682] -0.000374154*inputs[683] +0.000424075*inputs[684] +0.032525*inputs[685] -0.0122359*inputs[686] +0.000424081*inputs[687] -0.0122359*inputs[688] -0.0122359*inputs[689] +0.000424079*inputs[690] +0.022278*inputs[691] +0.000424081*inputs[692] +0.000424075*inputs[693] +0.00442392*inputs[694] +0.00497554*inputs[695] +0.00497554*inputs[696] +0.000424081*inputs[697] -0.0314154*inputs[698] -0.00901183*inputs[699] +0.00497554*inputs[700] +0.00497554*inputs[701] -0.0151192*inputs[702] +0.000424081*inputs[703] +0.000424081*inputs[704] +0.000424076*inputs[705] +0.000424078*inputs[706] +0.000424081*inputs[707] -0.0241953*inputs[708] +0.00042408*inputs[709] +0.000424081*inputs[710] +0.000424081*inputs[711] +0.000424081*inputs[712] -0.0151192*inputs[713] -0.0151192*inputs[714] -0.0151192*inputs[715] -0.00781601*inputs[716] +0.000424081*inputs[717] -0.0209933*inputs[718] -0.0209933*inputs[719] +0.000424072*inputs[720] +0.000424081*inputs[721] +0.000424081*inputs[722] +0.000424081*inputs[723] +0.00042408*inputs[724] +0.00432034*inputs[725] -0.00614682*inputs[726] -0.0209242*inputs[727] +0.0150069*inputs[728] +0.0150069*inputs[729] +0.0150069*inputs[730] -0.0209242*inputs[731] -0.0110334*inputs[732] +0.000424077*inputs[733] +0.000424075*inputs[734] +0.000424078*inputs[735] -0.0046167*inputs[736] -0.0046167*inputs[737] -0.0046167*inputs[738] -0.0046167*inputs[739] +0.00716455*inputs[740] -0.0209242*inputs[741] +0.0230947*inputs[742] +0.000424077*inputs[743] +0.00042408*inputs[744] +0.000424081*inputs[745] +0.00476899*inputs[746] +0.000424077*inputs[747] -0.0107628*inputs[748] +0.000424081*inputs[749] +0.000424077*inputs[750] -0.0236126*inputs[751] -0.0236126*inputs[752] +0.00042408*inputs[753] +0.000424081*inputs[754] +0.000424078*inputs[755] -0.0115941*inputs[756] +0.000424076*inputs[757] -0.0378274*inputs[758] +0.00476894*inputs[759] +0.00476894*inputs[760] +0.0136675*inputs[761] +0.0136675*inputs[762] +0.0136675*inputs[763] -0.0110334*inputs[764] +0.000424079*inputs[765] +0.00042408*inputs[766] +0.000424075*inputs[767] +0.00042408*inputs[768] +0.000424081*inputs[769] +0.000908055*inputs[770] +0.000424081*inputs[771] -0.00912556*inputs[772] +0.0211525*inputs[773] -0.00912556*inputs[774] -0.00931906*inputs[775] +0.0050225*inputs[776] -0.00912556*inputs[777] +0.0050225*inputs[778] +0.0050225*inputs[779] -0.000374154*inputs[780] -0.00797834*inputs[781] +0.035041*inputs[782] +0.000908055*inputs[783] +0.000424081*inputs[784] -0.026738*inputs[785] +0.00482777*inputs[786] +0.00482776*inputs[787] +0.00482776*inputs[788] +0.00482776*inputs[789] +0.0161381*inputs[790] -0.00774134*inputs[791] +0.000424081*inputs[792] +0.00042408*inputs[793] +0.0189455*inputs[794] +0.000424081*inputs[795] +0.0122304*inputs[796] +0.0122304*inputs[797] +0.000424081*inputs[798] +0.00042408*inputs[799] +0.00042408*inputs[800] +0.0142457*inputs[801] +0.0142457*inputs[802] +0.000424072*inputs[803] +0.00442392*inputs[804] +0.00042408*inputs[805] +0.0192111*inputs[806] -0.0129255*inputs[807] -0.0335062*inputs[808] +0.000424078*inputs[809] -0.0129255*inputs[810] +0.00748348*inputs[811] +0.0074835*inputs[812] +0.00748348*inputs[813] +0.00290587*inputs[814] +0.00290587*inputs[815] -0.00912556*inputs[816] +0.00042408*inputs[817] -0.00960219*inputs[818] +0.000424081*inputs[819] -0.00897149*inputs[820] -0.00897149*inputs[821] -0.0287678*inputs[822] +0.000424081*inputs[823] +0.000424081*inputs[824] -0.0331126*inputs[825] +0.0258731*inputs[826] +0.0258731*inputs[827] +0.00042408*inputs[828] +0.000424081*inputs[829] +0.000424077*inputs[830] +0.000424078*inputs[831] +0.000424081*inputs[832] -0.0141143*inputs[833] -0.0141143*inputs[834] -0.0141143*inputs[835] +0.00042408*inputs[836] +0.000424081*inputs[837] +0.00042408*inputs[838] +0.00192216*inputs[839] +0.00192212*inputs[840] +0.00192211*inputs[841] -0.0133506*inputs[842] +0.00042408*inputs[843] +0.000424075*inputs[844] -0.0295811*inputs[845] +0.0264273*inputs[846] +0.00985025*inputs[847] +0.00042408*inputs[848] -0.0116591*inputs[849] +0.0397683*inputs[850] -0.00720572*inputs[851] -0.00720573*inputs[852] -0.00720572*inputs[853] +0.000424081*inputs[854] -0.0196785*inputs[855] +0.000424081*inputs[856] +0.000424081*inputs[857] +0.000424081*inputs[858] +0.00366835*inputs[859] +0.00366835*inputs[860] +0.000424081*inputs[861] +0.0115397*inputs[862] +0.00576986*inputs[863] +0.0115397*inputs[864] +0.000424078*inputs[865] +0.0281983*inputs[866] +0.003281*inputs[867] +0.00328102*inputs[868] -0.00783381*inputs[869] +0.000424072*inputs[870] +0.000424081*inputs[871] +0.00042408*inputs[872] +0.0394674*inputs[873] +0.00042408*inputs[874] +0.00042408*inputs[875] -0.0133142*inputs[876] -0.0133142*inputs[877] +0.020167*inputs[878] +0.0157423*inputs[879] +0.0228558*inputs[880] +0.0228558*inputs[881] +0.00042408*inputs[882] +0.0297649*inputs[883] +0.0297649*inputs[884] +0.021572*inputs[885] +0.023224*inputs[886] +0.000424072*inputs[887] +0.000424081*inputs[888] +0.00042408*inputs[889] +0.000424081*inputs[890] -0.0175529*inputs[891] +0.000424081*inputs[892] +0.00042408*inputs[893] -0.0159593*inputs[894] -0.00520271*inputs[895] +0.000424079*inputs[896] -0.00520271*inputs[897] +0.000424081*inputs[898] +0.000424079*inputs[899] +0.000424076*inputs[900] -0.0379413*inputs[901] -0.0378271*inputs[902] +0.00042408*inputs[903] -0.0117163*inputs[904] -0.00571243*inputs[905] +0.000424081*inputs[906] +0.000424075*inputs[907] -0.0240584*inputs[908] -0.0240584*inputs[909] +0.000424077*inputs[910] +0.000424079*inputs[911] +0.00042408*inputs[912] +0.000424081*inputs[913] +0.00042408*inputs[914] +0.000424076*inputs[915] -0.0177673*inputs[916] -0.0177674*inputs[917] -0.0177673*inputs[918] +0.000424073*inputs[919] +0.000424081*inputs[920] +0.000424081*inputs[921] +0.018*inputs[922] +0.018*inputs[923] -0.00522774*inputs[924] -0.00522774*inputs[925] -0.00522774*inputs[926] +0.000424078*inputs[927] +0.000424081*inputs[928] +0.000424081*inputs[929] -0.00869609*inputs[930] -0.00869608*inputs[931] -0.00869608*inputs[932] -0.0037838*inputs[933] -0.0037838*inputs[934] +0.000424081*inputs[935] +0.0206973*inputs[936] +0.0206973*inputs[937] +0.000424081*inputs[938] +0.000424077*inputs[939] -0.00797834*inputs[940] -0.0242837*inputs[941] -0.0159401*inputs[942] -0.0311584*inputs[943] -0.00881597*inputs[944] -0.00881598*inputs[945] +0.000424081*inputs[946] +0.000424081*inputs[947] +0.0022485*inputs[948] +0.0022485*inputs[949] -0.0292679*inputs[950] +0.0272216*inputs[951] +0.000424081*inputs[952] +0.000424081*inputs[953] +0.000424077*inputs[954] +0.000424081*inputs[955] +0.000424075*inputs[956] +0.00042408*inputs[957] +0.000424081*inputs[958] +0.000424081*inputs[959] +0.000424081*inputs[960] +0.000424081*inputs[961] +0.000424076*inputs[962] +0.000424081*inputs[963] -0.0223883*inputs[964] -0.0188496*inputs[965] -0.0280752*inputs[966] -0.0185887*inputs[967] +0.000424081*inputs[968] +0.000424079*inputs[969] -0.00776726*inputs[970] -0.00776726*inputs[971] +0.0274418*inputs[972] +0.0274418*inputs[973] +0.000424078*inputs[974] -0.0223884*inputs[975] -0.00349332*inputs[976] +0.0271335*inputs[977] -0.00349332*inputs[978] +0.0268814*inputs[979] +0.0273962*inputs[980] +0.0110459*inputs[981] +0.00042408*inputs[982] +0.00922811*inputs[983] +0.00922811*inputs[984] -0.00901183*inputs[985] +0.00336217*inputs[986] +0.0116453*inputs[987] +0.0256668*inputs[988] +0.000424078*inputs[989] +0.000424081*inputs[990] +0.00826835*inputs[991] +0.00826835*inputs[992] +0.00826835*inputs[993] -0.0174217*inputs[994] +0.000424076*inputs[995] +0.000424081*inputs[996] +0.013938*inputs[997] +0.0174931*inputs[998] +0.00688482*inputs[999] +0.00042408*inputs[1000] -0.0245382*inputs[1001] +0.000424081*inputs[1002] +0.0106992*inputs[1003] +0.0106992*inputs[1004] +0.000424081*inputs[1005] +0.000424079*inputs[1006] +0.000424078*inputs[1007] +0.000424079*inputs[1008] +0.000424081*inputs[1009] +0.0291542*inputs[1010] -0.0392935*inputs[1011] +0.0271335*inputs[1012] +0.000424077*inputs[1013] -0.0122523*inputs[1014] +0.000424081*inputs[1015] +0.000424081*inputs[1016] -0.00817242*inputs[1017] -0.00817242*inputs[1018] +0.000424081*inputs[1019] +0.0295374*inputs[1020] +0.000424081*inputs[1021] -0.00532504*inputs[1022] +0.000424081*inputs[1023] -0.0122523*inputs[1024] +0.00042408*inputs[1025] -0.0122523*inputs[1026] +0.0160641*inputs[1027] +0.000424081*inputs[1028] -0.00770375*inputs[1029] +0.0296177*inputs[1030] +0.0165845*inputs[1031] +0.0165846*inputs[1032] -0.00770375*inputs[1033] +0.000424074*inputs[1034] +0.000424081*inputs[1035] -0.00770375*inputs[1036] -0.00784341*inputs[1037] +0.0109122*inputs[1038] +0.000424081*inputs[1039] +0.0223062*inputs[1040] +0.000424081*inputs[1041] +0.00704509*inputs[1042] +0.00704509*inputs[1043] -0.0113047*inputs[1044] -0.0113047*inputs[1045] -0.0113047*inputs[1046] -0.00784339*inputs[1047] -0.00784339*inputs[1048] +0.00883531*inputs[1049] -0.0195524*inputs[1050] +0.014898*inputs[1051] +0.014898*inputs[1052] +0.000424072*inputs[1053] +0.0327804*inputs[1054] +0.00042408*inputs[1055] +0.00042408*inputs[1056] +0.0425179*inputs[1057] +0.000424081*inputs[1058] +0.000424077*inputs[1059] +0.000424081*inputs[1060] +0.0113129*inputs[1061] -0.00642496*inputs[1062] +0.00603585*inputs[1063] +0.00603585*inputs[1064] +0.00042408*inputs[1065] +0.000424079*inputs[1066] +0.0183799*inputs[1067] +0.0183799*inputs[1068] +0.00716455*inputs[1069] +0.00175024*inputs[1070] -0.0275657*inputs[1071] +0.0113129*inputs[1072] +0.000424081*inputs[1073] +0.000424081*inputs[1074] +0.000424081*inputs[1075] +0.000424081*inputs[1076] +0.00378605*inputs[1077] +0.00378604*inputs[1078] +0.00378605*inputs[1079] -0.00316179*inputs[1080] -0.00316179*inputs[1081] +0.000424081*inputs[1082] +0.000424078*inputs[1083] +0.00815474*inputs[1084] +0.00042408*inputs[1085] +0.000424081*inputs[1086] +0.000424073*inputs[1087] +0.000424081*inputs[1088] +0.000424074*inputs[1089] -0.00770375*inputs[1090] -0.0310901*inputs[1091] +0.000424078*inputs[1092] -0.0141304*inputs[1093] +0.000424079*inputs[1094] +0.0248409*inputs[1095] -0.0253666*inputs[1096] +0.000424077*inputs[1097] -0.0171172*inputs[1098] +0.000424076*inputs[1099] +0.0161285*inputs[1100] +0.0358533*inputs[1101] -0.0394668*inputs[1102] -0.0072473*inputs[1103] -0.0080868*inputs[1104] +0.0149446*inputs[1105] +0.000424081*inputs[1106] -0.0072473*inputs[1107] +0.000424071*inputs[1108] -0.0168295*inputs[1109] +0.000424081*inputs[1110] +0.0112226*inputs[1111] +0.000424078*inputs[1112] -0.0193422*inputs[1113] +0.00735655*inputs[1114] +0.0168806*inputs[1115] -0.0168295*inputs[1116] -0.00731176*inputs[1117] -0.0263992*inputs[1118] +0.000424078*inputs[1119] +0.000424079*inputs[1120] +0.00582791*inputs[1121] +0.0111186*inputs[1122] -0.0326089*inputs[1123] +0.000424079*inputs[1124] -0.00226957*inputs[1125] +0.00735655*inputs[1126] -0.00797405*inputs[1127] +0.000424081*inputs[1128] -0.00731176*inputs[1129] +0.000424078*inputs[1130] +0.000424081*inputs[1131] +0.000424081*inputs[1132] +0.000424081*inputs[1133] +0.000424081*inputs[1134] +0.00590792*inputs[1135] +0.000424074*inputs[1136] -0.0158416*inputs[1137] +0.0346746*inputs[1138] +0.00735655*inputs[1139] +0.000424081*inputs[1140] -0.00521501*inputs[1141] -0.00521501*inputs[1142] +0.00042408*inputs[1143] +0.00735655*inputs[1144] +0.000424081*inputs[1145] +0.000424078*inputs[1146] +0.000424076*inputs[1147] +0.00042408*inputs[1148] +0.000424081*inputs[1149] -0.0208553*inputs[1150] +0.000424079*inputs[1151] +0.00042408*inputs[1152] +0.000424081*inputs[1153] +0.00582791*inputs[1154] +0.0058279*inputs[1155] +0.000424074*inputs[1156] +0.01504*inputs[1157] -0.0111685*inputs[1158] -0.0111685*inputs[1159] +0.020543*inputs[1160] -0.0102757*inputs[1161] -0.0251248*inputs[1162] +0.000424078*inputs[1163] +0.000424076*inputs[1164] -0.0102757*inputs[1165] +0.0293027*inputs[1166] +0.000424077*inputs[1167] +0.000424072*inputs[1168] -0.0111685*inputs[1169] +0.00790287*inputs[1170] +0.000424081*inputs[1171] +0.0110275*inputs[1172] +0.00042408*inputs[1173] +0.00790287*inputs[1174] +0.00042408*inputs[1175] +0.000424079*inputs[1176] +0.000424081*inputs[1177] +0.01504*inputs[1178] +0.000424078*inputs[1179] +0.000424079*inputs[1180] -0.0117212*inputs[1181] +0.000424081*inputs[1182] +0.000424081*inputs[1183] +0.000424081*inputs[1184] +0.000424081*inputs[1185] +0.000424077*inputs[1186] +0.00735655*inputs[1187] +0.00349971*inputs[1188] +0.00349971*inputs[1189] +0.00923864*inputs[1190] -0.0117212*inputs[1191] -0.0117212*inputs[1192] +0.0144845*inputs[1193] -0.0117212*inputs[1194] +0.00042408*inputs[1195] +0.000424075*inputs[1196] +0.00923864*inputs[1197] +0.0168783*inputs[1198] +0.0168783*inputs[1199] -0.0347431*inputs[1200] +0.000424081*inputs[1201] +0.000424081*inputs[1202] +0.000424073*inputs[1203] +0.000424075*inputs[1204] -0.00773482*inputs[1205] +0.0117911*inputs[1206] +0.000424078*inputs[1207] +0.000424074*inputs[1208] +0.000424081*inputs[1209] +0.000424077*inputs[1210] +0.00042408*inputs[1211] -0.00225696*inputs[1212] +0.0251877*inputs[1213] -0.00225696*inputs[1214] +0.00042408*inputs[1215] +0.000424074*inputs[1216] +0.000424075*inputs[1217] -0.00773482*inputs[1218] +0.0269824*inputs[1219] +0.000424081*inputs[1220] -0.00800194*inputs[1221] -0.00800194*inputs[1222] -0.00800194*inputs[1223] -0.00800194*inputs[1224] +0.0100381*inputs[1225] +0.000424079*inputs[1226] +0.000424081*inputs[1227] -0.00331822*inputs[1228] +0.000424081*inputs[1229] +0.00846751*inputs[1230] +0.00432347*inputs[1231] +0.00432346*inputs[1232] -0.00412465*inputs[1233] -0.00412465*inputs[1234] -0.00307238*inputs[1235] +0.000385667*inputs[1236] +0.000424077*inputs[1237] +0.000385666*inputs[1238] +0.000385667*inputs[1239] -0.00307238*inputs[1240] -0.00331822*inputs[1241] -0.011888*inputs[1242] +0.00042408*inputs[1243] -0.0187513*inputs[1244] -0.0187513*inputs[1245] +0.000424075*inputs[1246] +0.000424076*inputs[1247] -0.018086*inputs[1248] -0.018086*inputs[1249] +0.00042408*inputs[1250] +0.000424081*inputs[1251] +0.000424081*inputs[1252] +0.00184755*inputs[1253] +0.000424081*inputs[1254] +0.000424076*inputs[1255] +0.000424077*inputs[1256] +0.00042408*inputs[1257] +0.00659347*inputs[1258] +0.000424081*inputs[1259] +0.00659347*inputs[1260] +0.0307191*inputs[1261] +0.00042408*inputs[1262] +0.00042408*inputs[1263] +0.00184755*inputs[1264] +0.000424078*inputs[1265] +0.00184755*inputs[1266] +0.0103535*inputs[1267] +0.00042408*inputs[1268] -0.0172667*inputs[1269] +0.000424081*inputs[1270] +0.000424079*inputs[1271] +0.0214459*inputs[1272] +0.000424081*inputs[1273] +0.000424081*inputs[1274] +0.000424077*inputs[1275] +0.00042408*inputs[1276] +0.00500905*inputs[1277] +0.00042408*inputs[1278] -0.0303839*inputs[1279] +0.000424076*inputs[1280] -0.0135053*inputs[1281] +0.000424077*inputs[1282] +0.000424081*inputs[1283] -0.016871*inputs[1284] +0.000424081*inputs[1285] +0.00500905*inputs[1286] +0.00500905*inputs[1287] +0.00500905*inputs[1288] +0.00042408*inputs[1289] +0.0380015*inputs[1290] +0.0287651*inputs[1291] +0.000424081*inputs[1292] +0.0215462*inputs[1293] +0.00042408*inputs[1294] -0.0393318*inputs[1295] -0.000615782*inputs[1296] -0.00061578*inputs[1297] +0.000424079*inputs[1298] +0.00782668*inputs[1299] +0.000424081*inputs[1300] +0.00452597*inputs[1301] -0.0122853*inputs[1302] +0.00454883*inputs[1303] +0.000424081*inputs[1304] +0.000424075*inputs[1305] +0.0319478*inputs[1306] +0.00042408*inputs[1307] +0.000424081*inputs[1308] -0.00184259*inputs[1309] +0.000424076*inputs[1310] +0.00452596*inputs[1311] +0.00452597*inputs[1312] +0.00454883*inputs[1313] +0.00452597*inputs[1314] +0.00452596*inputs[1315] +0.000424081*inputs[1316] +0.000424075*inputs[1317] -0.00338798*inputs[1318] +0.000424077*inputs[1319] +0.000424081*inputs[1320] +0.000424079*inputs[1321] -0.0194504*inputs[1322] -0.0107248*inputs[1323] -0.0107248*inputs[1324] -0.0104168*inputs[1325] -0.00987479*inputs[1326] -0.00987479*inputs[1327] -0.00987479*inputs[1328] -0.00987479*inputs[1329] -0.0164149*inputs[1330] -0.0238448*inputs[1331] -0.0033578*inputs[1332] -0.0033578*inputs[1333] +0.000424081*inputs[1334] +0.000424078*inputs[1335] +0.000424081*inputs[1336] +0.0171394*inputs[1337] +0.000424081*inputs[1338] +0.00815474*inputs[1339] +0.0292761*inputs[1340] +0.00042408*inputs[1341] +0.00457695*inputs[1342] +0.00457697*inputs[1343] +0.000424077*inputs[1344] +0.00042408*inputs[1345] +0.000424079*inputs[1346] +0.000424081*inputs[1347] +0.00454883*inputs[1348] -0.00750915*inputs[1349] +0.0275563*inputs[1350] +0.0113122*inputs[1351] +0.0113121*inputs[1352] -0.00338798*inputs[1353] -0.0124729*inputs[1354] +0.00843277*inputs[1355] +0.00843277*inputs[1356] +0.00843277*inputs[1357] -0.00750915*inputs[1358] -0.00750916*inputs[1359] -0.0138419*inputs[1360] -0.00338798*inputs[1361] +0.000424081*inputs[1362] +0.00432347*inputs[1363] +0.00716455*inputs[1364] -0.0112492*inputs[1365] +0.00432347*inputs[1366] +0.00042408*inputs[1367] -0.0112492*inputs[1368] -0.0112492*inputs[1369] +0.000424074*inputs[1370] +0.00432347*inputs[1371] +0.000424077*inputs[1372] +0.00853854*inputs[1373] +0.000424078*inputs[1374] +0.016848*inputs[1375] +0.0117426*inputs[1376] +0.00540239*inputs[1377] +0.00540238*inputs[1378] +0.00540239*inputs[1379] +0.000424076*inputs[1380] +0.00893264*inputs[1381] +0.000424081*inputs[1382] +0.0379143*inputs[1383] +0.000424081*inputs[1384] +0.000424081*inputs[1385] +0.00853854*inputs[1386] +0.00845097*inputs[1387] +0.00845097*inputs[1388] +0.00845097*inputs[1389] +0.000424081*inputs[1390] +0.000424074*inputs[1391] +0.000424097*inputs[1392] -0.00338798*inputs[1393] +0.000424097*inputs[1394] -0.0204557*inputs[1395] 
		combinations[3] = 0.0132276 +0.0555831*inputs[0] +0.0575919*inputs[1] +0.232305*inputs[2] +0.160269*inputs[3] +0.087385*inputs[4] +0.0776545*inputs[5] +0.0451658*inputs[6] +0.00253234*inputs[7] +0.0294148*inputs[8] -0.00100407*inputs[9] +0.00247076*inputs[10] +0.106009*inputs[11] +0.0261716*inputs[12] +0.109871*inputs[13] +0.0451377*inputs[14] +0.12344*inputs[15] -0.00356863*inputs[16] +0.0108187*inputs[17] +0.104329*inputs[18] +0.00167289*inputs[19] +0.0406049*inputs[20] -0.0852618*inputs[21] +0.0292553*inputs[22] +0.032616*inputs[23] +0.124767*inputs[24] -0.00867073*inputs[25] -0.066346*inputs[26] -0.00966936*inputs[27] -0.00981816*inputs[28] -0.04173*inputs[29] -0.00925047*inputs[30] -0.0733647*inputs[31] -0.036389*inputs[32] +0.0964659*inputs[33] -0.0537257*inputs[34] +0.108622*inputs[35] +0.0697536*inputs[36] +0.0401357*inputs[37] +0.0128661*inputs[38] -0.113702*inputs[39] -0.0579964*inputs[40] +0.128943*inputs[41] +0.0585434*inputs[42] +0.0421111*inputs[43] -0.0120611*inputs[44] +0.0154323*inputs[45] -0.0355864*inputs[46] +0.0740077*inputs[47] -0.0412501*inputs[48] -0.00447939*inputs[49] +0.103143*inputs[50] +0.010654*inputs[51] -0.0178364*inputs[52] -0.0166085*inputs[53] +0.00644528*inputs[54] -0.0622706*inputs[55] -0.0260443*inputs[56] +0.0307661*inputs[57] -0.109339*inputs[58] -0.0050118*inputs[59] +0.0249576*inputs[60] -0.11679*inputs[61] -0.0380255*inputs[62] -0.0240558*inputs[63] -0.0419659*inputs[64] -0.0223769*inputs[65] +0.0712785*inputs[66] -0.0629695*inputs[67] +0.0220518*inputs[68] -0.0109132*inputs[69] -0.0116286*inputs[70] +0.022675*inputs[71] -0.00250386*inputs[72] +0.0272967*inputs[73] -0.0869689*inputs[74] +0.00707407*inputs[75] +0.0260072*inputs[76] -0.0365549*inputs[77] -0.180937*inputs[78] +0.0241148*inputs[79] -0.00628498*inputs[80] +0.013421*inputs[81] +0.0768494*inputs[82] -0.000624297*inputs[83] +0.0143061*inputs[84] -0.0271423*inputs[85] +0.0426002*inputs[86] -0.0432654*inputs[87] -0.0258429*inputs[88] +0.0128018*inputs[89] -0.0121833*inputs[90] +0.0304553*inputs[91] -0.0541718*inputs[92] -0.0661161*inputs[93] +0.0569472*inputs[94] -0.064765*inputs[95] -0.0139255*inputs[96] -0.0398515*inputs[97] -0.00996587*inputs[98] +0.00749488*inputs[99] -0.00591904*inputs[100] -0.00580534*inputs[101] +0.00909505*inputs[102] -0.0639586*inputs[103] -0.022843*inputs[104] -4.45959e-05*inputs[105] +0.0167969*inputs[106] +0.0128951*inputs[107] +0.0158956*inputs[108] -0.00443455*inputs[109] +0.0492761*inputs[110] +0.0605778*inputs[111] +0.0246951*inputs[112] -0.00869694*inputs[113] +0.074488*inputs[114] -0.0406503*inputs[115] -0.0528309*inputs[116] +0.00189506*inputs[117] +0.0460356*inputs[118] -0.0177522*inputs[119] -0.0165864*inputs[120] -0.00277933*inputs[121] -0.0574891*inputs[122] -0.00200159*inputs[123] -0.0088349*inputs[124] -0.0169585*inputs[125] -0.00470554*inputs[126] -0.0113666*inputs[127] -0.065835*inputs[128] +0.0248889*inputs[129] -0.00488957*inputs[130] +0.0379428*inputs[131] +0.0255885*inputs[132] +0.0894701*inputs[133] -0.0178256*inputs[134] -0.0234451*inputs[135] -0.00935439*inputs[136] -0.0471183*inputs[137] +0.00717726*inputs[138] +0.00926518*inputs[139] +0.00625023*inputs[140] +0.0280661*inputs[141] +0.0205231*inputs[142] -0.10622*inputs[143] +0.0105679*inputs[144] -0.0345193*inputs[145] -0.00839018*inputs[146] +0.00827386*inputs[147] -0.00725181*inputs[148] -0.00123431*inputs[149] -0.0161227*inputs[150] +0.0626865*inputs[151] -0.0541865*inputs[152] -0.0383809*inputs[153] +0.0118387*inputs[154] -0.00157694*inputs[155] -0.0233313*inputs[156] -0.0086134*inputs[157] -0.0602001*inputs[158] +0.0154157*inputs[159] +0.0347934*inputs[160] -0.0109957*inputs[161] -0.0178983*inputs[162] -0.00518743*inputs[163] -0.0350031*inputs[164] -0.00541447*inputs[165] -0.0101706*inputs[166] -0.0364179*inputs[167] -0.0247894*inputs[168] +0.0538624*inputs[169] -0.0116335*inputs[170] -0.0198559*inputs[171] -0.0305251*inputs[172] -0.0198791*inputs[173] -0.00416312*inputs[174] +0.00609742*inputs[175] -0.028371*inputs[176] +0.0119114*inputs[177] -0.0329616*inputs[178] +0.00887407*inputs[179] -0.0357238*inputs[180] +0.00380569*inputs[181] -0.0511287*inputs[182] -0.0065124*inputs[183] -0.0837781*inputs[184] +0.0186951*inputs[185] -0.0377653*inputs[186] +0.0171432*inputs[187] +0.00641454*inputs[188] -0.0236229*inputs[189] +0.073387*inputs[190] +0.0451937*inputs[191] +0.0109154*inputs[192] +0.0196366*inputs[193] -0.00413144*inputs[194] -0.00355153*inputs[195] -0.00410982*inputs[196] -0.00735371*inputs[197] +0.0245567*inputs[198] +0.0363936*inputs[199] +0.00662131*inputs[200] +0.0544149*inputs[201] +0.0226322*inputs[202] -0.0151982*inputs[203] -0.0026232*inputs[204] +0.00692572*inputs[205] +0.0330068*inputs[206] -0.00218581*inputs[207] -0.0229095*inputs[208] +0.0522584*inputs[209] -0.0580834*inputs[210] +0.00816151*inputs[211] -0.000937258*inputs[212] +0.00868046*inputs[213] +0.0223211*inputs[214] -0.00311724*inputs[215] -0.014461*inputs[216] +0.0061412*inputs[217] +0.0388322*inputs[218] -0.00293721*inputs[219] -0.0208323*inputs[220] +0.0178937*inputs[221] -0.00743395*inputs[222] -0.0470765*inputs[223] +0.00920939*inputs[224] -0.0467834*inputs[225] +0.016375*inputs[226] -0.0101579*inputs[227] -0.0272467*inputs[228] -0.0615814*inputs[229] +0.027877*inputs[230] -0.0137034*inputs[231] +0.00104843*inputs[232] -0.00583599*inputs[233] -0.00655949*inputs[234] -0.00461235*inputs[235] -0.0231371*inputs[236] +0.00440344*inputs[237] +0.0169404*inputs[238] -0.0163334*inputs[239] +0.0301566*inputs[240] +0.0206047*inputs[241] -0.00718537*inputs[242] +0.056838*inputs[243] +0.0206159*inputs[244] -0.0479104*inputs[245] +0.0100412*inputs[246] +0.0205371*inputs[247] -0.0237307*inputs[248] -0.0561843*inputs[249] -0.0139067*inputs[250] -0.0324697*inputs[251] +0.0145022*inputs[252] +0.020752*inputs[253] +0.00749921*inputs[254] +0.0336492*inputs[255] +0.0424172*inputs[256] +0.0761662*inputs[257] +0.0164663*inputs[258] +0.0367735*inputs[259] -0.0210214*inputs[260] -0.0138572*inputs[261] -0.0517498*inputs[262] +0.0319406*inputs[263] +0.00895936*inputs[264] -0.0192691*inputs[265] -0.0179308*inputs[266] -0.00595145*inputs[267] +0.0140947*inputs[268] -0.0145524*inputs[269] +0.0366617*inputs[270] +0.0638077*inputs[271] -0.00473067*inputs[272] +0.000594567*inputs[273] -0.00862669*inputs[274] +0.0306903*inputs[275] -0.0368169*inputs[276] -0.0437067*inputs[277] -0.00329959*inputs[278] -0.0522818*inputs[279] +0.0818675*inputs[280] -0.00425529*inputs[281] +0.00590421*inputs[282] -0.00581286*inputs[283] +0.00326278*inputs[284] -0.0175433*inputs[285] -0.0141918*inputs[286] +0.0654359*inputs[287] +0.00173359*inputs[288] -0.00878031*inputs[289] -0.000312339*inputs[290] -0.00459241*inputs[291] -0.00508315*inputs[292] -0.0031163*inputs[293] -0.00489284*inputs[294] +0.0162904*inputs[295] +0.0173778*inputs[296] -0.0160132*inputs[297] -0.0015415*inputs[298] -0.00965905*inputs[299] -0.00107126*inputs[300] -0.0084935*inputs[301] -0.00886626*inputs[302] +0.0165897*inputs[303] -0.00371578*inputs[304] +0.00624213*inputs[305] -0.027417*inputs[306] -0.00724411*inputs[307] -0.00191505*inputs[308] -0.0133755*inputs[309] +0.000757527*inputs[310] -0.0168988*inputs[311] +0.0146274*inputs[312] -0.0178594*inputs[313] -0.0330468*inputs[314] +0.0033447*inputs[315] -0.0148319*inputs[316] -0.0239815*inputs[317] -0.00825199*inputs[318] -0.0317932*inputs[319] +0.0335921*inputs[320] +0.00715743*inputs[321] -0.000561464*inputs[322] +0.0067822*inputs[323] +0.0104203*inputs[324] -0.025842*inputs[325] -0.00072528*inputs[326] -0.0104462*inputs[327] +0.0174056*inputs[328] -0.013435*inputs[329] -0.0116259*inputs[330] -0.0143712*inputs[331] +0.01081*inputs[332] -0.000513308*inputs[333] +0.0262736*inputs[334] -0.0179707*inputs[335] -0.0169989*inputs[336] +0.0279249*inputs[337] -0.00822151*inputs[338] -0.00216845*inputs[339] +0.0181704*inputs[340] -0.0319885*inputs[341] -0.0276415*inputs[342] -0.016271*inputs[343] -0.0149783*inputs[344] -0.00566795*inputs[345] -0.0151973*inputs[346] -0.0537797*inputs[347] -0.0314311*inputs[348] -0.0143609*inputs[349] +0.0193974*inputs[350] -0.00072528*inputs[351] -0.010045*inputs[352] +0.0289955*inputs[353] -0.0015502*inputs[354] -0.00994595*inputs[355] +0.0126233*inputs[356] +0.0119771*inputs[357] +0.013195*inputs[358] +0.02341*inputs[359] -0.00830641*inputs[360] -0.00896449*inputs[361] -0.0107998*inputs[362] +0.018262*inputs[363] -0.00332582*inputs[364] -0.0173177*inputs[365] +0.0071235*inputs[366] -0.00625746*inputs[367] -0.00878746*inputs[368] -0.0247658*inputs[369] +0.0436789*inputs[370] -0.00453946*inputs[371] -0.00236221*inputs[372] -0.0652258*inputs[373] -0.017583*inputs[374] -0.00072528*inputs[375] -0.015577*inputs[376] +0.0377175*inputs[377] -0.0417347*inputs[378] -0.0291938*inputs[379] -0.011413*inputs[380] -0.00497254*inputs[381] +0.00382037*inputs[382] -0.0174354*inputs[383] -0.0394089*inputs[384] +0.00926955*inputs[385] -0.0113989*inputs[386] -0.0041986*inputs[387] +0.000471485*inputs[388] +0.0144456*inputs[389] +0.0023161*inputs[390] -0.00072528*inputs[391] +0.0128519*inputs[392] +0.0233555*inputs[393] -0.00072528*inputs[394] +0.0147004*inputs[395] +0.00316251*inputs[396] -0.0447328*inputs[397] -0.0322474*inputs[398] -0.0412866*inputs[399] -0.0157987*inputs[400] -0.000725279*inputs[401] -0.00696729*inputs[402] -0.0130991*inputs[403] +0.00345476*inputs[404] +0.04969*inputs[405] -0.00222682*inputs[406] +0.0144087*inputs[407] +0.0288247*inputs[408] +0.0226697*inputs[409] -0.0148172*inputs[410] +0.00982743*inputs[411] -0.0145023*inputs[412] +0.00316299*inputs[413] -0.00961503*inputs[414] +0.023134*inputs[415] -0.0395417*inputs[416] +0.0259624*inputs[417] -0.0249207*inputs[418] +0.00584129*inputs[419] -0.0117788*inputs[420] -0.0446104*inputs[421] -0.0061544*inputs[422] +0.00725057*inputs[423] -0.00283221*inputs[424] -0.00072528*inputs[425] +0.00471612*inputs[426] -0.0155429*inputs[427] +0.00398274*inputs[428] -0.000110571*inputs[429] -0.000725277*inputs[430] -0.022529*inputs[431] +0.00914114*inputs[432] -0.0031118*inputs[433] +0.0151389*inputs[434] -0.00222184*inputs[435] +0.0164912*inputs[436] +0.0124197*inputs[437] -0.00536613*inputs[438] +0.0183822*inputs[439] -0.0200548*inputs[440] -0.0296817*inputs[441] -0.00622871*inputs[442] -0.0184733*inputs[443] -0.00072528*inputs[444] -0.0182394*inputs[445] -0.00526542*inputs[446] -0.0231828*inputs[447] -0.000112956*inputs[448] -0.000591888*inputs[449] +0.0197778*inputs[450] +0.0349268*inputs[451] -0.00317101*inputs[452] -0.0325205*inputs[453] +0.00500169*inputs[454] -0.0268813*inputs[455] -0.0354796*inputs[456] -0.013774*inputs[457] +0.0568389*inputs[458] -0.0497899*inputs[459] -0.0292691*inputs[460] +0.00413878*inputs[461] -0.00348813*inputs[462] -0.0205285*inputs[463] -0.0354937*inputs[464] -0.0541149*inputs[465] +0.000315723*inputs[466] -0.0104562*inputs[467] -0.0131849*inputs[468] -0.00589627*inputs[469] -0.0161107*inputs[470] -0.0215368*inputs[471] +0.0107292*inputs[472] -0.00883809*inputs[473] +0.0180286*inputs[474] +0.0180286*inputs[475] -0.0133981*inputs[476] -0.0253088*inputs[477] +0.0100392*inputs[478] -0.0107573*inputs[479] +0.00517351*inputs[480] +0.0396279*inputs[481] -0.0246512*inputs[482] -0.0209265*inputs[483] -0.000591892*inputs[484] -0.00820073*inputs[485] -0.0244472*inputs[486] -0.0220315*inputs[487] -0.0104825*inputs[488] -0.0173388*inputs[489] +0.00803191*inputs[490] -0.023781*inputs[491] -0.0115614*inputs[492] -0.0309605*inputs[493] -0.0107363*inputs[494] -0.00743824*inputs[495] -0.00743824*inputs[496] -0.0111827*inputs[497] +0.0683646*inputs[498] -0.00624032*inputs[499] -0.0209265*inputs[500] -0.00883809*inputs[501] -0.000591891*inputs[502] -0.000591892*inputs[503] -0.0268209*inputs[504] +0.0376334*inputs[505] +0.0197748*inputs[506] +0.0143056*inputs[507] -0.000591892*inputs[508] -0.00385768*inputs[509] +0.00198846*inputs[510] -0.0513818*inputs[511] -0.0179769*inputs[512] +0.0060691*inputs[513] -0.000591891*inputs[514] +0.0286379*inputs[515] -0.000591892*inputs[516] -0.0122719*inputs[517] -0.013774*inputs[518] -0.000591892*inputs[519] +0.00198846*inputs[520] +0.00624876*inputs[521] -0.0180577*inputs[522] +0.00239297*inputs[523] -0.00555811*inputs[524] -0.00555811*inputs[525] +0.00714509*inputs[526] -0.014455*inputs[527] -0.000591892*inputs[528] +0.00295447*inputs[529] -0.00846752*inputs[530] +0.0205808*inputs[531] +0.0107779*inputs[532] +0.0108774*inputs[533] -0.0169966*inputs[534] -0.0255368*inputs[535] -0.0143844*inputs[536] -0.0255193*inputs[537] +0.0108774*inputs[538] +0.011579*inputs[539] -0.000591888*inputs[540] -0.01178*inputs[541] -0.0131478*inputs[542] +0.00218797*inputs[543] -0.000591892*inputs[544] -0.00846753*inputs[545] -0.00141951*inputs[546] +0.00580487*inputs[547] +0.0201264*inputs[548] +0.005995*inputs[549] +0.00268835*inputs[550] +0.0135915*inputs[551] -0.00380497*inputs[552] -0.0151186*inputs[553] +0.00261503*inputs[554] -0.000591892*inputs[555] +0.00756954*inputs[556] +0.00408269*inputs[557] +0.0129571*inputs[558] -0.0405181*inputs[559] +0.00210827*inputs[560] +0.00590144*inputs[561] +0.00956901*inputs[562] +0.0360582*inputs[563] +0.0472042*inputs[564] -0.00059189*inputs[565] -0.00659798*inputs[566] -0.0109455*inputs[567] -0.000591892*inputs[568] +0.0151735*inputs[569] +0.00936449*inputs[570] -0.000591892*inputs[571] -0.0121792*inputs[572] -0.0338954*inputs[573] -0.000591892*inputs[574] +0.0264839*inputs[575] -0.000591891*inputs[576] -0.0395954*inputs[577] +0.0136217*inputs[578] -0.0249248*inputs[579] -0.00493937*inputs[580] -0.00152207*inputs[581] -0.000591888*inputs[582] -0.00846752*inputs[583] -0.00583273*inputs[584] -0.0204119*inputs[585] +0.0244008*inputs[586] -0.000591892*inputs[587] +0.029127*inputs[588] -0.000591892*inputs[589] -0.00858042*inputs[590] -0.000418331*inputs[591] -0.000591892*inputs[592] -0.0678747*inputs[593] -0.000920173*inputs[594] +0.027526*inputs[595] -0.00451803*inputs[596] -0.0315882*inputs[597] -0.0051031*inputs[598] -0.00917984*inputs[599] +0.0197404*inputs[600] +0.035767*inputs[601] -0.0120031*inputs[602] -0.0115816*inputs[603] -0.0182275*inputs[604] -0.000591892*inputs[605] +0.00478198*inputs[606] -0.0051031*inputs[607] +0.00839228*inputs[608] +0.0180418*inputs[609] -0.000591891*inputs[610] +0.0117811*inputs[611] +0.0318299*inputs[612] -0.00820074*inputs[613] -0.000418325*inputs[614] -0.00700629*inputs[615] +0.00186841*inputs[616] -0.0375682*inputs[617] -0.0141116*inputs[618] -0.00152535*inputs[619] +0.0220526*inputs[620] -0.00431552*inputs[621] -0.0130544*inputs[622] +0.0220526*inputs[623] -0.000894129*inputs[624] +0.0125072*inputs[625] -0.0195932*inputs[626] -0.000591888*inputs[627] -0.000591892*inputs[628] -0.000591892*inputs[629] -0.0176158*inputs[630] -0.0172585*inputs[631] +0.00692121*inputs[632] -0.035675*inputs[633] -0.00967099*inputs[634] -0.00555811*inputs[635] -0.000591892*inputs[636] +0.0107987*inputs[637] -0.0223103*inputs[638] -0.000591892*inputs[639] -0.0365539*inputs[640] +0.0150734*inputs[641] +0.00339666*inputs[642] +0.0257054*inputs[643] -0.0365381*inputs[644] -0.0176158*inputs[645] -0.0565942*inputs[646] -0.0120381*inputs[647] -0.000418331*inputs[648] +0.00889229*inputs[649] +0.00274339*inputs[650] +0.0286466*inputs[651] -0.00827423*inputs[652] +0.00889229*inputs[653] -0.00827423*inputs[654] -0.00260714*inputs[655] +0.00889229*inputs[656] -0.000418327*inputs[657] -0.000418331*inputs[658] +0.00274337*inputs[659] -0.000418331*inputs[660] -0.000418328*inputs[661] -0.00406389*inputs[662] -0.000418327*inputs[663] +0.00630639*inputs[664] +0.00325426*inputs[665] -0.0380721*inputs[666] -0.0233185*inputs[667] -0.000418331*inputs[668] +0.024271*inputs[669] -0.00443583*inputs[670] -0.0243411*inputs[671] -0.000418331*inputs[672] -0.00406389*inputs[673] +0.00899345*inputs[674] +0.0138085*inputs[675] +0.00325426*inputs[676] -0.000418331*inputs[677] -0.000418328*inputs[678] -0.000418323*inputs[679] +0.00325426*inputs[680] -0.00260717*inputs[681] -0.000418327*inputs[682] +0.000399271*inputs[683] -0.00041833*inputs[684] -0.032175*inputs[685] +0.0120975*inputs[686] -0.000418329*inputs[687] +0.0120975*inputs[688] +0.0120975*inputs[689] -0.000418328*inputs[690] -0.0220369*inputs[691] -0.000418331*inputs[692] -0.000418331*inputs[693] -0.00443583*inputs[694] -0.00496006*inputs[695] -0.00496006*inputs[696] -0.000418331*inputs[697] +0.0309888*inputs[698] +0.00899345*inputs[699] -0.00496006*inputs[700] -0.00496006*inputs[701] +0.0149443*inputs[702] -0.000418328*inputs[703] -0.000418325*inputs[704] -0.000418327*inputs[705] -0.000418327*inputs[706] -0.000418331*inputs[707] +0.0238943*inputs[708] -0.00041833*inputs[709] -0.000418331*inputs[710] -0.00041833*inputs[711] -0.000418331*inputs[712] +0.0149443*inputs[713] +0.0149443*inputs[714] +0.0149443*inputs[715] +0.00777069*inputs[716] -0.000418327*inputs[717] +0.0207256*inputs[718] +0.0207256*inputs[719] -0.000418327*inputs[720] -0.000418324*inputs[721] -0.000418326*inputs[722] -0.000418331*inputs[723] -0.000418331*inputs[724] -0.00436824*inputs[725] +0.00618655*inputs[726] +0.0206446*inputs[727] -0.0148343*inputs[728] -0.0148343*inputs[729] -0.0148343*inputs[730] +0.0206446*inputs[731] +0.0109439*inputs[732] -0.000418331*inputs[733] -0.000418325*inputs[734] -0.000418331*inputs[735] +0.00459448*inputs[736] +0.00459448*inputs[737] +0.00459448*inputs[738] +0.00459448*inputs[739] -0.00712885*inputs[740] +0.0206446*inputs[741] -0.0228984*inputs[742] -0.000418331*inputs[743] -0.000418328*inputs[744] -0.000418331*inputs[745] -0.00476118*inputs[746] -0.000418331*inputs[747] +0.0106442*inputs[748] -0.000418331*inputs[749] -0.000418325*inputs[750] +0.0233184*inputs[751] +0.0233184*inputs[752] -0.000418324*inputs[753] -0.000418328*inputs[754] -0.000418327*inputs[755] +0.0115382*inputs[756] -0.00041833*inputs[757] +0.0373064*inputs[758] -0.00476119*inputs[759] -0.00476119*inputs[760] -0.0135361*inputs[761] -0.0135361*inputs[762] -0.0135361*inputs[763] +0.0109439*inputs[764] -0.000418331*inputs[765] -0.000418331*inputs[766] -0.000418331*inputs[767] -0.000418331*inputs[768] -0.000418331*inputs[769] -0.000926684*inputs[770] -0.000418331*inputs[771] +0.0090621*inputs[772] -0.0209519*inputs[773] +0.0090621*inputs[774] +0.00925099*inputs[775] -0.00503452*inputs[776] +0.0090621*inputs[777] -0.0050345*inputs[778] -0.00503451*inputs[779] +0.000399272*inputs[780] +0.00791756*inputs[781] -0.0345965*inputs[782] -0.000926678*inputs[783] -0.000418329*inputs[784] +0.0263999*inputs[785] -0.00483384*inputs[786] -0.00483384*inputs[787] -0.00483384*inputs[788] -0.00483384*inputs[789] -0.0160374*inputs[790] +0.00773112*inputs[791] -0.000418326*inputs[792] -0.000418327*inputs[793] -0.018812*inputs[794] -0.00041833*inputs[795] -0.0120744*inputs[796] -0.0120744*inputs[797] -0.000418331*inputs[798] -0.000418327*inputs[799] -0.000418331*inputs[800] -0.0141092*inputs[801] -0.0141092*inputs[802] -0.000418331*inputs[803] -0.00443582*inputs[804] -0.00041833*inputs[805] -0.0190514*inputs[806] +0.0128041*inputs[807] +0.0330926*inputs[808] -0.000418331*inputs[809] +0.0128041*inputs[810] -0.0074381*inputs[811] -0.0074381*inputs[812] -0.0074381*inputs[813] -0.00292391*inputs[814] -0.00292391*inputs[815] +0.0090621*inputs[816] -0.000418331*inputs[817] +0.00958291*inputs[818] -0.000418331*inputs[819] +0.00899704*inputs[820] +0.00899703*inputs[821] +0.0283701*inputs[822] -0.000418326*inputs[823] -0.000418331*inputs[824] +0.0325718*inputs[825] -0.0255368*inputs[826] -0.0255368*inputs[827] -0.00041833*inputs[828] -0.000418329*inputs[829] -0.000418331*inputs[830] -0.000418326*inputs[831] -0.000418331*inputs[832] +0.0139548*inputs[833] +0.0139548*inputs[834] +0.0139548*inputs[835] -0.000418323*inputs[836] -0.000418331*inputs[837] -0.000418331*inputs[838] -0.0019291*inputs[839] -0.00192911*inputs[840] -0.00192914*inputs[841] +0.0132761*inputs[842] -0.00041833*inputs[843] -0.000418326*inputs[844] +0.0291788*inputs[845] -0.0260575*inputs[846] -0.00987581*inputs[847] -0.000418331*inputs[848] +0.011579*inputs[849] -0.0391218*inputs[850] +0.00717767*inputs[851] +0.00717767*inputs[852] +0.00717766*inputs[853] -0.000418331*inputs[854] +0.0194875*inputs[855] -0.000418331*inputs[856] -0.000418324*inputs[857] -0.000418331*inputs[858] -0.00369081*inputs[859] -0.0036908*inputs[860] -0.000418326*inputs[861] -0.0114655*inputs[862] -0.00574844*inputs[863] -0.0114655*inputs[864] -0.000418324*inputs[865] -0.0277698*inputs[866] -0.0032821*inputs[867] -0.0032821*inputs[868] +0.00777981*inputs[869] -0.000418329*inputs[870] -0.000418324*inputs[871] -0.000418331*inputs[872] -0.0388854*inputs[873] -0.000418331*inputs[874] -0.000418331*inputs[875] +0.0131625*inputs[876] +0.0131625*inputs[877] -0.0200139*inputs[878] -0.015658*inputs[879] -0.0225949*inputs[880] -0.0225949*inputs[881] -0.000418328*inputs[882] -0.0293633*inputs[883] -0.0293633*inputs[884] -0.0213942*inputs[885] -0.0230263*inputs[886] -0.000418331*inputs[887] -0.000418327*inputs[888] -0.00041833*inputs[889] -0.00041833*inputs[890] +0.0174393*inputs[891] -0.000418331*inputs[892] -0.000418331*inputs[893] +0.0157936*inputs[894] +0.00523957*inputs[895] -0.00041833*inputs[896] +0.00523957*inputs[897] -0.000418328*inputs[898] -0.000418331*inputs[899] -0.000418326*inputs[900] +0.0374338*inputs[901] +0.0373065*inputs[902] -0.000418331*inputs[903] +0.0116518*inputs[904] +0.00573313*inputs[905] -0.000418331*inputs[906] -0.00041833*inputs[907] +0.0236681*inputs[908] +0.0236681*inputs[909] -0.00041833*inputs[910] -0.000418331*inputs[911] -0.00041833*inputs[912] -0.000418331*inputs[913] -0.000418331*inputs[914] -0.000418324*inputs[915] +0.0176164*inputs[916] +0.0176164*inputs[917] +0.0176164*inputs[918] -0.000418329*inputs[919] -0.000418331*inputs[920] -0.000418328*inputs[921] -0.0178225*inputs[922] -0.0178225*inputs[923] +0.00521952*inputs[924] +0.00521952*inputs[925] +0.00521952*inputs[926] -0.000418331*inputs[927] -0.000418331*inputs[928] -0.00041833*inputs[929] +0.00862355*inputs[930] +0.00862356*inputs[931] +0.00862356*inputs[932] +0.00381234*inputs[933] +0.00381234*inputs[934] -0.000418327*inputs[935] -0.020528*inputs[936] -0.020528*inputs[937] -0.000418328*inputs[938] -0.000418331*inputs[939] +0.00791756*inputs[940] +0.0239774*inputs[941] +0.0158081*inputs[942] +0.030786*inputs[943] +0.00876004*inputs[944] +0.00876004*inputs[945] -0.000418331*inputs[946] -0.000418331*inputs[947] -0.00229187*inputs[948] -0.00229188*inputs[949] +0.0288671*inputs[950] -0.0269571*inputs[951] -0.000418325*inputs[952] -0.000418329*inputs[953] -0.000418331*inputs[954] -0.000418331*inputs[955] -0.000418327*inputs[956] -0.000418324*inputs[957] -0.000418326*inputs[958] -0.000418331*inputs[959] -0.00041833*inputs[960] -0.000418325*inputs[961] -0.000418326*inputs[962] -0.000418328*inputs[963] +0.0221106*inputs[964] +0.0186686*inputs[965] +0.0277242*inputs[966] +0.0184587*inputs[967] -0.00041833*inputs[968] -0.000418329*inputs[969] +0.00772803*inputs[970] +0.00772803*inputs[971] -0.0270861*inputs[972] -0.027086*inputs[973] -0.00041833*inputs[974] +0.0221106*inputs[975] +0.00351105*inputs[976] -0.0268737*inputs[977] +0.00351105*inputs[978] -0.0266062*inputs[979] -0.0269879*inputs[980] -0.0110047*inputs[981] -0.000418331*inputs[982] -0.00919165*inputs[983] -0.00919166*inputs[984] +0.00899345*inputs[985] -0.00337926*inputs[986] -0.0115506*inputs[987] -0.0254022*inputs[988] -0.000418325*inputs[989] -0.00041833*inputs[990] -0.00823246*inputs[991] -0.00823245*inputs[992] -0.00823246*inputs[993] +0.0172758*inputs[994] -0.000418331*inputs[995] -0.000418331*inputs[996] -0.0138444*inputs[997] -0.0174312*inputs[998] -0.00688368*inputs[999] -0.000418331*inputs[1000] +0.024302*inputs[1001] -0.000418331*inputs[1002] -0.01063*inputs[1003] -0.01063*inputs[1004] -0.000418331*inputs[1005] -0.000418331*inputs[1006] -0.000418328*inputs[1007] -0.000418331*inputs[1008] -0.00041833*inputs[1009] -0.0287902*inputs[1010] +0.0385893*inputs[1011] -0.0268737*inputs[1012] -0.000418331*inputs[1013] +0.012111*inputs[1014] -0.000418331*inputs[1015] -0.000418331*inputs[1016] +0.00814584*inputs[1017] +0.00814584*inputs[1018] -0.000418329*inputs[1019] -0.0291285*inputs[1020] -0.000418331*inputs[1021] +0.00535095*inputs[1022] -0.000418325*inputs[1023] +0.012111*inputs[1024] -0.000418328*inputs[1025] +0.012111*inputs[1026] -0.0159238*inputs[1027] -0.000418331*inputs[1028] +0.00763722*inputs[1029] -0.0291614*inputs[1030] -0.0164064*inputs[1031] -0.0164064*inputs[1032] +0.00763722*inputs[1033] -0.000418324*inputs[1034] -0.000418331*inputs[1035] +0.00763722*inputs[1036] +0.00775219*inputs[1037] -0.0108466*inputs[1038] -0.000418328*inputs[1039] -0.0221276*inputs[1040] -0.000418327*inputs[1041] -0.00701816*inputs[1042] -0.00701816*inputs[1043] +0.0112199*inputs[1044] +0.0112199*inputs[1045] +0.0112199*inputs[1046] +0.00775219*inputs[1047] +0.00775219*inputs[1048] -0.00884348*inputs[1049] +0.0193937*inputs[1050] -0.0147575*inputs[1051] -0.0147575*inputs[1052] -0.000418327*inputs[1053] -0.0324634*inputs[1054] -0.000418325*inputs[1055] -0.000418326*inputs[1056] -0.0420161*inputs[1057] -0.00041833*inputs[1058] -0.000418331*inputs[1059] -0.000418331*inputs[1060] -0.011219*inputs[1061] +0.00641737*inputs[1062] -0.00596796*inputs[1063] -0.00596794*inputs[1064] -0.000418331*inputs[1065] -0.000418324*inputs[1066] -0.0182186*inputs[1067] -0.0182186*inputs[1068] -0.00712885*inputs[1069] -0.00177094*inputs[1070] +0.0272057*inputs[1071] -0.011219*inputs[1072] -0.000418331*inputs[1073] -0.00041833*inputs[1074] -0.000418329*inputs[1075] -0.000418331*inputs[1076] -0.00381071*inputs[1077] -0.00381072*inputs[1078] -0.0038107*inputs[1079] +0.00318401*inputs[1080] +0.00318401*inputs[1081] -0.000418331*inputs[1082] -0.000418331*inputs[1083] -0.00810466*inputs[1084] -0.000418331*inputs[1085] -0.000418331*inputs[1086] -0.000418331*inputs[1087] -0.000418331*inputs[1088] -0.000418329*inputs[1089] +0.00763723*inputs[1090] +0.0306638*inputs[1091] -0.000418331*inputs[1092] +0.0141315*inputs[1093] -0.00041833*inputs[1094] -0.0244817*inputs[1095] +0.0250864*inputs[1096] -0.00041833*inputs[1097] +0.0168759*inputs[1098] -0.00041833*inputs[1099] -0.0159524*inputs[1100] -0.0353235*inputs[1101] +0.0389528*inputs[1102] +0.00719598*inputs[1103] +0.00806516*inputs[1104] -0.014854*inputs[1105] -0.000418331*inputs[1106] +0.00719598*inputs[1107] -0.000418325*inputs[1108] +0.0166217*inputs[1109] -0.00041833*inputs[1110] -0.0111735*inputs[1111] -0.000418331*inputs[1112] +0.0191567*inputs[1113] -0.00729932*inputs[1114] -0.0167744*inputs[1115] +0.0166217*inputs[1116] +0.00729595*inputs[1117] +0.0260721*inputs[1118] -0.00041833*inputs[1119] -0.000418331*inputs[1120] -0.00576605*inputs[1121] -0.0110854*inputs[1122] +0.032227*inputs[1123] -0.000418331*inputs[1124] +0.00232399*inputs[1125] -0.00729933*inputs[1126] +0.00795533*inputs[1127] -0.000418329*inputs[1128] +0.00729595*inputs[1129] -0.000418324*inputs[1130] -0.000418326*inputs[1131] -0.000418325*inputs[1132] -0.000418331*inputs[1133] -0.000418331*inputs[1134] -0.00590081*inputs[1135] -0.000418331*inputs[1136] +0.0156823*inputs[1137] -0.0343045*inputs[1138] -0.00729933*inputs[1139] -0.000418326*inputs[1140] +0.00523282*inputs[1141] +0.00523282*inputs[1142] -0.000418331*inputs[1143] -0.00729932*inputs[1144] -0.000418331*inputs[1145] -0.000418325*inputs[1146] -0.000418331*inputs[1147] -0.000418328*inputs[1148] -0.000418328*inputs[1149] +0.0206597*inputs[1150] -0.000418331*inputs[1151] -0.00041833*inputs[1152] -0.000418325*inputs[1153] -0.00576603*inputs[1154] -0.00576603*inputs[1155] -0.000418324*inputs[1156] -0.0148726*inputs[1157] +0.0110003*inputs[1158] +0.0110003*inputs[1159] -0.0202868*inputs[1160] +0.0102015*inputs[1161] +0.0248028*inputs[1162] -0.00041833*inputs[1163] -0.000418328*inputs[1164] +0.0102015*inputs[1165] -0.0288496*inputs[1166] -0.000418331*inputs[1167] -0.000418324*inputs[1168] +0.0110003*inputs[1169] -0.0078728*inputs[1170] -0.000418326*inputs[1171] -0.0109663*inputs[1172] -0.000418331*inputs[1173] -0.00787276*inputs[1174] -0.000418327*inputs[1175] -0.00041833*inputs[1176] -0.000418331*inputs[1177] -0.0148726*inputs[1178] -0.000418331*inputs[1179] -0.000418331*inputs[1180] +0.0115892*inputs[1181] -0.000418328*inputs[1182] -0.00041833*inputs[1183] -0.000418326*inputs[1184] -0.000418328*inputs[1185] -0.000418331*inputs[1186] -0.00729932*inputs[1187] -0.00355803*inputs[1188] -0.00355803*inputs[1189] -0.00917984*inputs[1190] +0.0115892*inputs[1191] +0.0115892*inputs[1192] -0.0143615*inputs[1193] +0.0115892*inputs[1194] -0.00041833*inputs[1195] -0.00041833*inputs[1196] -0.00917984*inputs[1197] -0.0167532*inputs[1198] -0.0167533*inputs[1199] +0.0342891*inputs[1200] -0.000418331*inputs[1201] -0.000418331*inputs[1202] -0.00041833*inputs[1203] -0.000418331*inputs[1204] +0.007663*inputs[1205] -0.0117102*inputs[1206] -0.000418331*inputs[1207] -0.00041833*inputs[1208] -0.000418326*inputs[1209] -0.000418326*inputs[1210] -0.000418329*inputs[1211] +0.00227766*inputs[1212] -0.0249733*inputs[1213] +0.00227766*inputs[1214] -0.00041833*inputs[1215] -0.000418324*inputs[1216] -0.000418328*inputs[1217] +0.007663*inputs[1218] -0.0266597*inputs[1219] -0.000418331*inputs[1220] +0.00793433*inputs[1221] +0.00793433*inputs[1222] +0.00793433*inputs[1223] +0.00793433*inputs[1224] -0.00999078*inputs[1225] -0.000418331*inputs[1226] -0.000418331*inputs[1227] +0.00328972*inputs[1228] -0.000418331*inputs[1229] -0.00841116*inputs[1230] -0.00431785*inputs[1231] -0.00431785*inputs[1232] +0.00411468*inputs[1233] +0.00411469*inputs[1234] +0.00310119*inputs[1235] -0.000376001*inputs[1236] -0.000418331*inputs[1237] -0.000375995*inputs[1238] -0.000375993*inputs[1239] +0.00310119*inputs[1240] +0.00328973*inputs[1241] +0.0118379*inputs[1242] -0.000418331*inputs[1243] +0.01851*inputs[1244] +0.01851*inputs[1245] -0.000418331*inputs[1246] -0.000418329*inputs[1247] +0.0178659*inputs[1248] +0.0178659*inputs[1249] -0.000418331*inputs[1250] -0.000418326*inputs[1251] -0.000418327*inputs[1252] -0.0018624*inputs[1253] -0.000418331*inputs[1254] -0.00041833*inputs[1255] -0.000418331*inputs[1256] -0.000418331*inputs[1257] -0.00656346*inputs[1258] -0.000418331*inputs[1259] -0.00656346*inputs[1260] -0.0303834*inputs[1261] -0.000418331*inputs[1262] -0.000418324*inputs[1263] -0.0018624*inputs[1264] -0.00041833*inputs[1265] -0.0018624*inputs[1266] -0.0102392*inputs[1267] -0.000418331*inputs[1268] +0.017071*inputs[1269] -0.000418331*inputs[1270] -0.000418329*inputs[1271] -0.0212857*inputs[1272] -0.000418329*inputs[1273] -0.000418331*inputs[1274] -0.000418331*inputs[1275] -0.000418331*inputs[1276] -0.00498426*inputs[1277] -0.000418331*inputs[1278] +0.0299628*inputs[1279] -0.000418325*inputs[1280] +0.0133824*inputs[1281] -0.000418331*inputs[1282] -0.000418328*inputs[1283] +0.0166537*inputs[1284] -0.000418324*inputs[1285] -0.00498426*inputs[1286] -0.00498425*inputs[1287] -0.00498425*inputs[1288] -0.000418329*inputs[1289] -0.0374642*inputs[1290] -0.028434*inputs[1291] -0.000418331*inputs[1292] -0.0213758*inputs[1293] -0.000418328*inputs[1294] +0.03879*inputs[1295] +0.000645301*inputs[1296] +0.0006453*inputs[1297] -0.000418329*inputs[1298] -0.00782637*inputs[1299] -0.000418329*inputs[1300] -0.00451216*inputs[1301] +0.0121787*inputs[1302] -0.00454781*inputs[1303] -0.000418331*inputs[1304] -0.000418331*inputs[1305] -0.0316032*inputs[1306] -0.000418331*inputs[1307] -0.000418331*inputs[1308] +0.0018773*inputs[1309] -0.000418331*inputs[1310] -0.00451216*inputs[1311] -0.00451216*inputs[1312] -0.00454781*inputs[1313] -0.00451216*inputs[1314] -0.00451216*inputs[1315] -0.000418328*inputs[1316] -0.000418331*inputs[1317] +0.00339835*inputs[1318] -0.000418331*inputs[1319] -0.000418331*inputs[1320] -0.000418331*inputs[1321] +0.0192056*inputs[1322] +0.0106397*inputs[1323] +0.0106397*inputs[1324] +0.0103607*inputs[1325] +0.00976986*inputs[1326] +0.00976985*inputs[1327] +0.00976986*inputs[1328] +0.00976985*inputs[1329] +0.0162461*inputs[1330] +0.0235608*inputs[1331] +0.00339544*inputs[1332] +0.00339544*inputs[1333] -0.00041833*inputs[1334] -0.00041833*inputs[1335] -0.00041833*inputs[1336] -0.0170185*inputs[1337] -0.000418328*inputs[1338] -0.00810466*inputs[1339] -0.0290265*inputs[1340] -0.00041833*inputs[1341] -0.00458193*inputs[1342] -0.00458194*inputs[1343] -0.000418326*inputs[1344] -0.000418331*inputs[1345] -0.000418327*inputs[1346] -0.000418326*inputs[1347] -0.00454781*inputs[1348] +0.00744708*inputs[1349] -0.0272769*inputs[1350] -0.0111647*inputs[1351] -0.0111647*inputs[1352] +0.00339836*inputs[1353] +0.0124066*inputs[1354] -0.00837468*inputs[1355] -0.00837468*inputs[1356] -0.00837468*inputs[1357] +0.00744708*inputs[1358] +0.00744708*inputs[1359] +0.0137205*inputs[1360] +0.00339835*inputs[1361] -0.000418331*inputs[1362] -0.00431785*inputs[1363] -0.00712886*inputs[1364] +0.0111179*inputs[1365] -0.00431785*inputs[1366] -0.000418331*inputs[1367] +0.0111179*inputs[1368] +0.0111179*inputs[1369] -0.00041833*inputs[1370] -0.00431785*inputs[1371] -0.000418331*inputs[1372] -0.00851084*inputs[1373] -0.00041833*inputs[1374] -0.0167282*inputs[1375] -0.011694*inputs[1376] -0.00538555*inputs[1377] -0.00538559*inputs[1378] -0.00538557*inputs[1379] -0.000418331*inputs[1380] -0.00890805*inputs[1381] -0.000418331*inputs[1382] -0.0375194*inputs[1383] -0.000418327*inputs[1384] -0.000418328*inputs[1385] -0.00851083*inputs[1386] -0.00840242*inputs[1387] -0.00840244*inputs[1388] -0.00840242*inputs[1389] -0.000418327*inputs[1390] -0.000418326*inputs[1391] -0.000418333*inputs[1392] +0.00339836*inputs[1393] -0.000418333*inputs[1394] +0.0202558*inputs[1395] 
		combinations[4] = -0.0132761 -0.055881*inputs[0] -0.0577425*inputs[1] -0.233331*inputs[2] -0.161027*inputs[3] -0.0882543*inputs[4] -0.0780461*inputs[5] -0.0454392*inputs[6] -0.00258675*inputs[7] -0.0295201*inputs[8] +0.000966238*inputs[9] -0.00249451*inputs[10] -0.106545*inputs[11] -0.0263011*inputs[12] -0.11051*inputs[13] -0.0453663*inputs[14] -0.124167*inputs[15] +0.00352123*inputs[16] -0.0108611*inputs[17] -0.104726*inputs[18] -0.00169515*inputs[19] -0.0407721*inputs[20] +0.0856239*inputs[21] -0.0291524*inputs[22] -0.0327338*inputs[23] -0.125374*inputs[24] +0.00869686*inputs[25] +0.0666003*inputs[26] +0.00969847*inputs[27] +0.00985705*inputs[28] +0.0418693*inputs[29] +0.00922008*inputs[30] +0.073663*inputs[31] +0.0365107*inputs[32] -0.0968585*inputs[33] +0.0540251*inputs[34] -0.1091*inputs[35] -0.0700019*inputs[36] -0.0403496*inputs[37] -0.012997*inputs[38] +0.114008*inputs[39] +0.0582515*inputs[40] -0.129743*inputs[41] -0.0591231*inputs[42] -0.0423604*inputs[43] +0.0120538*inputs[44] -0.0154907*inputs[45] +0.0356931*inputs[46] -0.0743656*inputs[47] +0.0413422*inputs[48] +0.0044968*inputs[49] -0.103655*inputs[50] -0.0106846*inputs[51] +0.0178899*inputs[52] +0.0166533*inputs[53] -0.00646497*inputs[54] +0.0624207*inputs[55] +0.0261482*inputs[56] -0.0309053*inputs[57] +0.109726*inputs[58] +0.00503495*inputs[59] -0.0246977*inputs[60] +0.117279*inputs[61] +0.038125*inputs[62] +0.0241336*inputs[63] +0.0420636*inputs[64] +0.022438*inputs[65] -0.0716993*inputs[66] +0.0631877*inputs[67] -0.0221232*inputs[68] +0.0109669*inputs[69] +0.011674*inputs[70] -0.0228132*inputs[71] +0.00251178*inputs[72] -0.0274184*inputs[73] +0.0872727*inputs[74] -0.00708879*inputs[75] -0.0261572*inputs[76] +0.0366487*inputs[77] +0.181877*inputs[78] -0.0242176*inputs[79] +0.00632607*inputs[80] -0.0134729*inputs[81] -0.0772843*inputs[82] +0.000613832*inputs[83] -0.0143884*inputs[84] +0.0272314*inputs[85] -0.0427387*inputs[86] +0.0434666*inputs[87] +0.0259182*inputs[88] -0.0128423*inputs[89] +0.0122009*inputs[90] -0.030562*inputs[91] +0.054389*inputs[92] +0.0665448*inputs[93] -0.0572049*inputs[94] +0.0651186*inputs[95] +0.0139813*inputs[96] +0.0400349*inputs[97] +0.0100015*inputs[98] -0.00758122*inputs[99] +0.00593375*inputs[100] +0.0058358*inputs[101] -0.00907293*inputs[102] +0.0641543*inputs[103] +0.0229786*inputs[104] +3.00848e-05*inputs[105] -0.0168296*inputs[106] -0.0129343*inputs[107] -0.0159283*inputs[108] +0.00445425*inputs[109] -0.0494812*inputs[110] -0.0608058*inputs[111] -0.0248559*inputs[112] +0.00870091*inputs[113] -0.07489*inputs[114] +0.040789*inputs[115] +0.0529869*inputs[116] -0.00190263*inputs[117] -0.046216*inputs[118] +0.0177791*inputs[119] +0.0166246*inputs[120] +0.00273659*inputs[121] +0.0576721*inputs[122] +0.00200946*inputs[123] +0.00886851*inputs[124] +0.0170074*inputs[125] +0.00472543*inputs[126] +0.0113874*inputs[127] +0.0660922*inputs[128] -0.0249712*inputs[129] +0.00490779*inputs[130] -0.0380664*inputs[131] -0.0256552*inputs[132] -0.0899677*inputs[133] +0.0179145*inputs[134] +0.0235149*inputs[135] +0.00937362*inputs[136] +0.047244*inputs[137] -0.00719912*inputs[138] -0.00929037*inputs[139] -0.0062706*inputs[140] -0.0282167*inputs[141] -0.0206014*inputs[142] +0.106738*inputs[143] -0.0105591*inputs[144] +0.0346206*inputs[145] +0.00842279*inputs[146] -0.00830992*inputs[147] +0.00728722*inputs[148] +0.00125493*inputs[149] +0.0161643*inputs[150] -0.0629649*inputs[151] +0.0543706*inputs[152] +0.0384884*inputs[153] -0.011927*inputs[154] +0.00158984*inputs[155] +0.0233913*inputs[156] +0.00862763*inputs[157] +0.0603898*inputs[158] -0.015433*inputs[159] -0.0349093*inputs[160] +0.0110316*inputs[161] +0.0179481*inputs[162] +0.00519596*inputs[163] +0.0351041*inputs[164] +0.00541903*inputs[165] +0.010179*inputs[166] +0.0365271*inputs[167] +0.024868*inputs[168] -0.0540749*inputs[169] +0.011654*inputs[170] +0.0199225*inputs[171] +0.0306394*inputs[172] +0.0199351*inputs[173] +0.00414619*inputs[174] -0.00612175*inputs[175] +0.0284291*inputs[176] -0.011944*inputs[177] +0.0330639*inputs[178] -0.0088927*inputs[179] +0.0358577*inputs[180] -0.00379401*inputs[181] +0.0513172*inputs[182] +0.00652212*inputs[183] +0.0841629*inputs[184] -0.0187521*inputs[185] +0.0379194*inputs[186] -0.0171827*inputs[187] -0.00644333*inputs[188] +0.0236793*inputs[189] -0.0737268*inputs[190] -0.0453636*inputs[191] -0.0109528*inputs[192] -0.019694*inputs[193] +0.00415448*inputs[194] +0.00355182*inputs[195] +0.00410809*inputs[196] +0.0073687*inputs[197] -0.0246605*inputs[198] -0.0365244*inputs[199] -0.00665616*inputs[200] -0.0546359*inputs[201] -0.0227362*inputs[202] +0.0152664*inputs[203] +0.00262693*inputs[204] -0.00694415*inputs[205] -0.0331409*inputs[206] +0.00219124*inputs[207] +0.0229563*inputs[208] -0.0524802*inputs[209] +0.05827*inputs[210] -0.0081963*inputs[211] +0.000940692*inputs[212] -0.00872464*inputs[213] -0.0224735*inputs[214] +0.00312264*inputs[215] +0.0145106*inputs[216] -0.00616396*inputs[217] -0.0389537*inputs[218] +0.00295442*inputs[219] +0.0209257*inputs[220] -0.0180176*inputs[221] +0.00744173*inputs[222] +0.0472279*inputs[223] -0.00925147*inputs[224] +0.0469259*inputs[225] -0.0164448*inputs[226] +0.0101755*inputs[227] +0.0273115*inputs[228] +0.0618073*inputs[229] -0.0279576*inputs[230] +0.0137307*inputs[231] -0.0010532*inputs[232] +0.00584584*inputs[233] +0.0065709*inputs[234] +0.0046117*inputs[235] +0.0232237*inputs[236] -0.00441507*inputs[237] -0.0169775*inputs[238] +0.0163664*inputs[239] -0.0302298*inputs[240] -0.0206934*inputs[241] +0.00720297*inputs[242] -0.0570854*inputs[243] -0.02066*inputs[244] +0.0480546*inputs[245] -0.0100791*inputs[246] -0.0206073*inputs[247] +0.0237875*inputs[248] +0.0563804*inputs[249] +0.0139809*inputs[250] +0.0326171*inputs[251] -0.0145308*inputs[252] -0.0208288*inputs[253] -0.00753722*inputs[254] -0.0337719*inputs[255] -0.0425594*inputs[256] -0.0765651*inputs[257] -0.0165055*inputs[258] -0.036913*inputs[259] +0.0210413*inputs[260] +0.013897*inputs[261] +0.0519146*inputs[262] -0.0320639*inputs[263] -0.00900871*inputs[264] +0.0192981*inputs[265] +0.0180031*inputs[266] +0.00596305*inputs[267] -0.0141404*inputs[268] +0.0145792*inputs[269] -0.0368189*inputs[270] -0.0640692*inputs[271] +0.00474025*inputs[272] -0.000602813*inputs[273] +0.0086366*inputs[274] -0.0307856*inputs[275] +0.0369012*inputs[276] +0.0438495*inputs[277] +0.0032942*inputs[278] +0.0524346*inputs[279] -0.0822471*inputs[280] +0.00424142*inputs[281] -0.00593041*inputs[282] +0.00581779*inputs[283] -0.00326658*inputs[284] +0.017608*inputs[285] +0.0142419*inputs[286] -0.065701*inputs[287] -0.00173095*inputs[288] +0.00879576*inputs[289] +0.000318939*inputs[290] +0.00459522*inputs[291] +0.00509803*inputs[292] +0.00311887*inputs[293] +0.00488671*inputs[294] -0.0163538*inputs[295] -0.0174226*inputs[296] +0.0160519*inputs[297] +0.00157924*inputs[298] +0.00967094*inputs[299] +0.00107154*inputs[300] +0.00851695*inputs[301] +0.00890134*inputs[302] -0.0166325*inputs[303] +0.00372753*inputs[304] -0.0062562*inputs[305] +0.027483*inputs[306] +0.00724675*inputs[307] +0.00192699*inputs[308] +0.0134324*inputs[309] -0.000751161*inputs[310] +0.0169471*inputs[311] -0.0147066*inputs[312] +0.0178886*inputs[313] +0.0331834*inputs[314] -0.00335151*inputs[315] +0.0148655*inputs[316] +0.0240509*inputs[317] +0.00829232*inputs[318] +0.0318987*inputs[319] -0.0337012*inputs[320] -0.00717327*inputs[321] +0.000563499*inputs[322] -0.00683161*inputs[323] -0.0104465*inputs[324] +0.0259124*inputs[325] +0.00072792*inputs[326] +0.0104649*inputs[327] -0.0174431*inputs[328] +0.0134877*inputs[329] +0.0116303*inputs[330] +0.0143967*inputs[331] -0.0108363*inputs[332] +0.000519551*inputs[333] -0.026378*inputs[334] +0.0180102*inputs[335] +0.0170317*inputs[336] -0.0280604*inputs[337] +0.00823845*inputs[338] +0.00217364*inputs[339] -0.0182342*inputs[340] +0.0321281*inputs[341] +0.0277098*inputs[342] +0.0163328*inputs[343] +0.0150063*inputs[344] +0.00568875*inputs[345] +0.0152259*inputs[346] +0.0539667*inputs[347] +0.0315435*inputs[348] +0.0144136*inputs[349] -0.0194418*inputs[350] +0.000727919*inputs[351] +0.0100617*inputs[352] -0.0290865*inputs[353] +0.00154882*inputs[354] +0.0099697*inputs[355] -0.01265*inputs[356] -0.0120088*inputs[357] -0.0132538*inputs[358] -0.0234958*inputs[359] +0.00832861*inputs[360] +0.00899884*inputs[361] +0.0108261*inputs[362] -0.0183013*inputs[363] +0.00333531*inputs[364] +0.0173829*inputs[365] -0.00714527*inputs[366] +0.00626662*inputs[367] +0.00879687*inputs[368] +0.0248455*inputs[369] -0.0438638*inputs[370] +0.00454544*inputs[371] +0.00236893*inputs[372] +0.0654963*inputs[373] +0.0176176*inputs[374] +0.000727919*inputs[375] +0.0156229*inputs[376] -0.0378576*inputs[377] +0.0418894*inputs[378] +0.0292713*inputs[379] +0.0114308*inputs[380] +0.00501418*inputs[381] -0.00382824*inputs[382] +0.0174719*inputs[383] +0.0395298*inputs[384] -0.00930516*inputs[385] +0.0114218*inputs[386] +0.00420104*inputs[387] -0.00048078*inputs[388] -0.0144871*inputs[389] -0.00232841*inputs[390] +0.000727919*inputs[391] -0.0128802*inputs[392] -0.023427*inputs[393] +0.000727919*inputs[394] -0.0147418*inputs[395] -0.00316947*inputs[396] +0.0449062*inputs[397] +0.0323191*inputs[398] +0.0414478*inputs[399] +0.0158366*inputs[400] +0.000727919*inputs[401] +0.00698367*inputs[402] +0.0131257*inputs[403] -0.00346758*inputs[404] -0.049882*inputs[405] +0.00223705*inputs[406] -0.0144431*inputs[407] -0.028923*inputs[408] -0.0227517*inputs[409] +0.0148667*inputs[410] -0.00983657*inputs[411] +0.0145313*inputs[412] -0.00317697*inputs[413] +0.00962972*inputs[414] -0.0231919*inputs[415] +0.0397216*inputs[416] -0.0260376*inputs[417] +0.024991*inputs[418] -0.00585346*inputs[419] +0.0117921*inputs[420] +0.0448002*inputs[421] +0.00616994*inputs[422] -0.00726796*inputs[423] +0.00284117*inputs[424] +0.000727919*inputs[425] -0.00471063*inputs[426] +0.0156019*inputs[427] -0.00398711*inputs[428] +0.000116886*inputs[429] +0.000727919*inputs[430] +0.0226004*inputs[431] -0.00917309*inputs[432] +0.00311379*inputs[433] -0.0151578*inputs[434] +0.00223176*inputs[435] -0.0165477*inputs[436] -0.0124547*inputs[437] +0.00535657*inputs[438] -0.0184343*inputs[439] +0.0200849*inputs[440] +0.029751*inputs[441] +0.00624766*inputs[442] +0.0185176*inputs[443] +0.000727919*inputs[444] +0.0182832*inputs[445] +0.00527247*inputs[446] +0.0232174*inputs[447] +0.000110541*inputs[448] +0.000594046*inputs[449] -0.0198482*inputs[450] -0.0350852*inputs[451] +0.00318093*inputs[452] +0.0326015*inputs[453] -0.0050149*inputs[454] +0.0269739*inputs[455] +0.0355784*inputs[456] +0.0138055*inputs[457] -0.05707*inputs[458] +0.0499712*inputs[459] +0.0293246*inputs[460] -0.00415949*inputs[461] +0.00349184*inputs[462] +0.0206401*inputs[463] +0.0355887*inputs[464] +0.0543594*inputs[465] -0.000304927*inputs[466] +0.0104807*inputs[467] +0.0132167*inputs[468] +0.00589812*inputs[469] +0.016145*inputs[470] +0.021588*inputs[471] -0.0107439*inputs[472] +0.0088688*inputs[473] -0.0180897*inputs[474] -0.0180897*inputs[475] +0.0134174*inputs[476] +0.0253736*inputs[477] -0.0100947*inputs[478] +0.0107652*inputs[479] -0.00517451*inputs[480] -0.039784*inputs[481] +0.0247383*inputs[482] +0.0210141*inputs[483] +0.000594046*inputs[484] +0.00821125*inputs[485] +0.024528*inputs[486] +0.022084*inputs[487] +0.0105165*inputs[488] +0.0173762*inputs[489] -0.00807085*inputs[490] +0.0238291*inputs[491] +0.0115892*inputs[492] +0.0310387*inputs[493] +0.0107642*inputs[494] +0.00745028*inputs[495] +0.00745027*inputs[496] +0.0111982*inputs[497] -0.0686775*inputs[498] +0.0062507*inputs[499] +0.0210141*inputs[500] +0.0088688*inputs[501] +0.000594046*inputs[502] +0.000594046*inputs[503] +0.0268991*inputs[504] -0.0377827*inputs[505] -0.0198491*inputs[506] -0.0143675*inputs[507] +0.000594046*inputs[508] +0.00385654*inputs[509] -0.00199407*inputs[510] +0.0515595*inputs[511] +0.0180234*inputs[512] -0.00606345*inputs[513] +0.000594046*inputs[514] -0.0287523*inputs[515] +0.000594047*inputs[516] +0.0123382*inputs[517] +0.0138055*inputs[518] +0.000594046*inputs[519] -0.00199407*inputs[520] -0.0062607*inputs[521] +0.0180974*inputs[522] -0.0023851*inputs[523] +0.00556789*inputs[524] +0.00556789*inputs[525] -0.00719468*inputs[526] +0.0144854*inputs[527] +0.000594047*inputs[528] -0.00295766*inputs[529] +0.00848663*inputs[530] -0.0206151*inputs[531] -0.0107866*inputs[532] -0.0109078*inputs[533] +0.0170349*inputs[534] +0.0256269*inputs[535] +0.0144072*inputs[536] +0.0256064*inputs[537] -0.0109078*inputs[538] -0.0116007*inputs[539] +0.000594047*inputs[540] +0.0118176*inputs[541] +0.0131909*inputs[542] -0.00218361*inputs[543] +0.000594046*inputs[544] +0.00848663*inputs[545] +0.00141884*inputs[546] -0.00581761*inputs[547] -0.0202014*inputs[548] -0.00600698*inputs[549] -0.00269606*inputs[550] -0.0136603*inputs[551] +0.00380903*inputs[552] +0.0151579*inputs[553] -0.00261588*inputs[554] +0.000594046*inputs[555] -0.00759338*inputs[556] -0.00407556*inputs[557] -0.0129845*inputs[558] +0.0406532*inputs[559] -0.00210526*inputs[560] -0.00591104*inputs[561] -0.00961006*inputs[562] -0.0361925*inputs[563] -0.0473997*inputs[564] +0.000594046*inputs[565] +0.00660375*inputs[566] +0.0109559*inputs[567] +0.000594046*inputs[568] -0.0152095*inputs[569] -0.0093768*inputs[570] +0.000594046*inputs[571] +0.0122105*inputs[572] +0.0339882*inputs[573] +0.000594046*inputs[574] -0.0265369*inputs[575] +0.000594046*inputs[576] +0.0397113*inputs[577] -0.0136688*inputs[578] +0.0249918*inputs[579] +0.00494624*inputs[580] +0.00151318*inputs[581] +0.000594046*inputs[582] +0.00848663*inputs[583] +0.00583383*inputs[584] +0.020476*inputs[585] -0.0245097*inputs[586] +0.000594046*inputs[587] -0.0292291*inputs[588] +0.000594046*inputs[589] +0.00859698*inputs[590] +0.000419843*inputs[591] +0.000594046*inputs[592] +0.068169*inputs[593] +0.000927088*inputs[594] -0.0276227*inputs[595] +0.00453196*inputs[596] +0.03169*inputs[597] +0.00509991*inputs[598] +0.00919567*inputs[599] -0.0197957*inputs[600] -0.0359192*inputs[601] +0.0120281*inputs[602] +0.0116163*inputs[603] +0.0182966*inputs[604] +0.000594046*inputs[605] -0.00478634*inputs[606] +0.00509993*inputs[607] -0.00839524*inputs[608] -0.0180673*inputs[609] +0.000594047*inputs[610] -0.0118172*inputs[611] -0.0319448*inputs[612] +0.00821126*inputs[613] +0.000419844*inputs[614] +0.00702684*inputs[615] -0.00188153*inputs[616] +0.0377018*inputs[617] +0.0141312*inputs[618] +0.00152043*inputs[619] -0.0221323*inputs[620] +0.00432168*inputs[621] +0.013069*inputs[622] -0.0221323*inputs[623] +0.000894216*inputs[624] -0.0125428*inputs[625] +0.0196476*inputs[626] +0.000594047*inputs[627] +0.000594046*inputs[628] +0.000594046*inputs[629] +0.0176849*inputs[630] +0.0173288*inputs[631] -0.00693432*inputs[632] +0.0357753*inputs[633] +0.0096917*inputs[634] +0.00556789*inputs[635] +0.000594046*inputs[636] -0.0108279*inputs[637] +0.0223881*inputs[638] +0.000594046*inputs[639] +0.0366785*inputs[640] -0.0151183*inputs[641] -0.0033972*inputs[642] -0.0258154*inputs[643] +0.0366273*inputs[644] +0.0176849*inputs[645] +0.0567991*inputs[646] +0.0120555*inputs[647] +0.000419849*inputs[648] -0.00891076*inputs[649] -0.00273283*inputs[650] -0.0287276*inputs[651] +0.00828339*inputs[652] -0.00891076*inputs[653] +0.00828339*inputs[654] +0.00260208*inputs[655] -0.00891076*inputs[656] +0.000419849*inputs[657] +0.000419842*inputs[658] -0.00273283*inputs[659] +0.000419847*inputs[660] +0.000419847*inputs[661] +0.00407641*inputs[662] +0.000419846*inputs[663] -0.00629061*inputs[664] -0.00325348*inputs[665] +0.0382377*inputs[666] +0.0233694*inputs[667] +0.000419848*inputs[668] -0.0243347*inputs[669] +0.00443272*inputs[670] +0.024391*inputs[671] +0.000419849*inputs[672] +0.00407639*inputs[673] -0.0089986*inputs[674] -0.013837*inputs[675] -0.00325348*inputs[676] +0.000419849*inputs[677] +0.000419848*inputs[678] +0.000419849*inputs[679] -0.00325348*inputs[680] +0.00260208*inputs[681] +0.000419849*inputs[682] -0.000392523*inputs[683] +0.000419849*inputs[684] +0.0322692*inputs[685] -0.0121346*inputs[686] +0.000419844*inputs[687] -0.0121346*inputs[688] -0.0121346*inputs[689] +0.000419849*inputs[690] +0.0221018*inputs[691] +0.000419849*inputs[692] +0.000419849*inputs[693] +0.00443272*inputs[694] +0.00496426*inputs[695] +0.00496425*inputs[696] +0.000419848*inputs[697] -0.0311033*inputs[698] -0.0089986*inputs[699] +0.00496426*inputs[700] +0.00496425*inputs[701] -0.0149912*inputs[702] +0.000419849*inputs[703] +0.000419849*inputs[704] +0.000419849*inputs[705] +0.000419849*inputs[706] +0.000419849*inputs[707] -0.0239752*inputs[708] +0.000419849*inputs[709] +0.000419849*inputs[710] +0.000419849*inputs[711] +0.000419843*inputs[712] -0.0149912*inputs[713] -0.0149913*inputs[714] -0.0149912*inputs[715] -0.00778288*inputs[716] +0.000419847*inputs[717] -0.0207975*inputs[718] -0.0207975*inputs[719] +0.000419849*inputs[720] +0.000419849*inputs[721] +0.000419849*inputs[722] +0.000419849*inputs[723] +0.000419847*inputs[724] +0.00435562*inputs[725] -0.00617617*inputs[726] -0.0207196*inputs[727] +0.0148804*inputs[728] +0.0148805*inputs[729] +0.0148804*inputs[730] -0.0207196*inputs[731] -0.0109681*inputs[732] +0.000419847*inputs[733] +0.000419848*inputs[734] +0.000419849*inputs[735] -0.00460051*inputs[736] -0.00460051*inputs[737] -0.00460051*inputs[738] -0.00460051*inputs[739] +0.00713847*inputs[740] -0.0207196*inputs[741] +0.0229514*inputs[742] +0.000419849*inputs[743] +0.000419849*inputs[744] +0.000419841*inputs[745] +0.00476333*inputs[746] +0.000419845*inputs[747] -0.0106758*inputs[748] +0.000419848*inputs[749] +0.000419849*inputs[750] -0.0233974*inputs[751] -0.0233974*inputs[752] +0.000419845*inputs[753] +0.000419844*inputs[754] +0.000419844*inputs[755] -0.0115535*inputs[756] +0.000419849*inputs[757] -0.0374462*inputs[758] +0.00476331*inputs[759] +0.00476331*inputs[760] +0.0135715*inputs[761] +0.0135715*inputs[762] +0.0135715*inputs[763] -0.0109681*inputs[764] +0.000419849*inputs[765] +0.000419849*inputs[766] +0.000419849*inputs[767] +0.000419845*inputs[768] +0.000419849*inputs[769] +0.000921596*inputs[770] +0.000419849*inputs[771] -0.00907923*inputs[772] +0.021006*inputs[773] -0.00907923*inputs[774] -0.00926939*inputs[775] +0.00503142*inputs[776] -0.00907923*inputs[777] +0.00503143*inputs[778] +0.00503143*inputs[779] -0.000392523*inputs[780] -0.00793393*inputs[781] +0.0347159*inputs[782] +0.00092161*inputs[783] +0.000419849*inputs[784] -0.0264906*inputs[785] +0.00483222*inputs[786] +0.00483224*inputs[787] +0.00483226*inputs[788] +0.00483222*inputs[789] +0.0160647*inputs[790] -0.00773406*inputs[791] +0.000419847*inputs[792] +0.000419844*inputs[793] +0.0188481*inputs[794] +0.000419846*inputs[795] +0.0121162*inputs[796] +0.0121162*inputs[797] +0.000419848*inputs[798] +0.000419849*inputs[799] +0.000419845*inputs[800] +0.0141458*inputs[801] +0.0141458*inputs[802] +0.000419849*inputs[803] +0.00443272*inputs[804] +0.000419849*inputs[805] +0.0190944*inputs[806] -0.0128368*inputs[807] -0.0332037*inputs[808] +0.000419842*inputs[809] -0.0128368*inputs[810] +0.0074504*inputs[811] +0.0074504*inputs[812] +0.0074504*inputs[813] +0.00291911*inputs[814] +0.00291916*inputs[815] -0.00907923*inputs[816] +0.000419849*inputs[817] -0.00958835*inputs[818] +0.000419849*inputs[819] -0.00899058*inputs[820] -0.00899058*inputs[821] -0.0284767*inputs[822] +0.000419849*inputs[823] +0.000419849*inputs[824] -0.0327164*inputs[825] +0.0256269*inputs[826] +0.0256269*inputs[827] +0.000419849*inputs[828] +0.000419849*inputs[829] +0.000419848*inputs[830] +0.000419849*inputs[831] +0.000419849*inputs[832] -0.0139976*inputs[833] -0.0139976*inputs[834] -0.0139976*inputs[835] +0.000419849*inputs[836] +0.000419847*inputs[837] +0.000419842*inputs[838] +0.00192721*inputs[839] +0.00192722*inputs[840] +0.00192721*inputs[841] -0.0132963*inputs[842] +0.000419848*inputs[843] +0.000419849*inputs[844] -0.0292867*inputs[845] +0.0261566*inputs[846] +0.00986956*inputs[847] +0.000419848*inputs[848] -0.0116007*inputs[849] +0.0392948*inputs[850] -0.00718538*inputs[851] -0.00718538*inputs[852] -0.00718538*inputs[853] +0.000419848*inputs[854] -0.0195389*inputs[855] +0.000419847*inputs[856] +0.000419845*inputs[857] +0.000419849*inputs[858] +0.00368461*inputs[859] +0.00368466*inputs[860] +0.000419849*inputs[861] +0.0114855*inputs[862] +0.00575398*inputs[863] +0.0114856*inputs[864] +0.000419843*inputs[865] +0.0278844*inputs[866] +0.00328194*inputs[867] +0.00328194*inputs[868] -0.00779433*inputs[869] +0.000419847*inputs[870] +0.000419849*inputs[871] +0.000419847*inputs[872] +0.0390411*inputs[873] +0.000419846*inputs[874] +0.000419847*inputs[875] -0.0132033*inputs[876] -0.0132033*inputs[877] +0.0200553*inputs[878] +0.0156808*inputs[879] +0.0226651*inputs[880] +0.0226651*inputs[881] +0.000419843*inputs[882] +0.0294711*inputs[883] +0.0294711*inputs[884] +0.0214423*inputs[885] +0.0230797*inputs[886] +0.000419845*inputs[887] +0.000419849*inputs[888] +0.000419842*inputs[889] +0.000419849*inputs[890] -0.0174701*inputs[891] +0.000419849*inputs[892] +0.000419849*inputs[893] -0.0158381*inputs[894] -0.00522982*inputs[895] +0.000419849*inputs[896] -0.00522983*inputs[897] +0.000419849*inputs[898] +0.000419849*inputs[899] +0.000419849*inputs[900] -0.0375699*inputs[901] -0.0374462*inputs[902] +0.000419849*inputs[903] -0.0116694*inputs[904] -0.00572782*inputs[905] +0.000419849*inputs[906] +0.000419849*inputs[907] -0.023773*inputs[908] -0.023773*inputs[909] +0.000419847*inputs[910] +0.000419847*inputs[911] +0.00041984*inputs[912] +0.000419847*inputs[913] +0.000419845*inputs[914] +0.000419849*inputs[915] -0.0176571*inputs[916] -0.0176571*inputs[917] -0.0176571*inputs[918] +0.000419849*inputs[919] +0.000419849*inputs[920] +0.000419845*inputs[921] +0.0178704*inputs[922] +0.0178704*inputs[923] -0.00522181*inputs[924] -0.00522181*inputs[925] -0.00522181*inputs[926] +0.000419842*inputs[927] +0.000419844*inputs[928] +0.000419849*inputs[929] -0.00864308*inputs[930] -0.00864308*inputs[931] -0.00864308*inputs[932] -0.00380474*inputs[933] -0.00380475*inputs[934] +0.000419844*inputs[935] +0.0205737*inputs[936] +0.0205737*inputs[937] +0.000419849*inputs[938] +0.000419841*inputs[939] -0.00793393*inputs[940] -0.0240596*inputs[941] -0.0158436*inputs[942] -0.0308861*inputs[943] -0.00877513*inputs[944] -0.00877513*inputs[945] +0.000419847*inputs[946] +0.000419849*inputs[947] +0.00228034*inputs[948] +0.00228034*inputs[949] -0.0289746*inputs[950] +0.0270283*inputs[951] +0.000419844*inputs[952] +0.000419849*inputs[953] +0.000419849*inputs[954] +0.000419849*inputs[955] +0.000419849*inputs[956] +0.000419847*inputs[957] +0.000419849*inputs[958] +0.000419849*inputs[959] +0.000419849*inputs[960] +0.000419848*inputs[961] +0.000419843*inputs[962] +0.000419845*inputs[963] -0.0221852*inputs[964] -0.0187174*inputs[965] -0.0278185*inputs[966] -0.0184939*inputs[967] +0.000419848*inputs[968] +0.000419849*inputs[969] -0.00773866*inputs[970] -0.00773866*inputs[971] +0.0271815*inputs[972] +0.0271815*inputs[973] +0.000419849*inputs[974] -0.0221851*inputs[975] -0.00350639*inputs[976] +0.0269437*inputs[977] -0.00350639*inputs[978] +0.02668*inputs[979] +0.0270975*inputs[980] +0.0110159*inputs[981] +0.000419849*inputs[982] +0.00920151*inputs[983] +0.00920154*inputs[984] -0.0089986*inputs[985] +0.00337486*inputs[986] +0.011576*inputs[987] +0.0254734*inputs[988] +0.000419849*inputs[989] +0.000419848*inputs[990] +0.00824216*inputs[991] +0.00824216*inputs[992] +0.00824216*inputs[993] -0.0173152*inputs[994] +0.000419849*inputs[995] +0.000419849*inputs[996] +0.0138698*inputs[997] +0.0174482*inputs[998] +0.00688404*inputs[999] +0.000419849*inputs[1000] -0.0243656*inputs[1001] +0.000419849*inputs[1002] +0.0106487*inputs[1003] +0.0106487*inputs[1004] +0.000419844*inputs[1005] +0.000419849*inputs[1006] +0.000419849*inputs[1007] +0.000419844*inputs[1008] +0.000419849*inputs[1009] +0.0288878*inputs[1010] -0.0387784*inputs[1011] +0.0269437*inputs[1012] +0.000419844*inputs[1013] -0.0121489*inputs[1014] +0.000419847*inputs[1015] +0.000419844*inputs[1016] -0.00815309*inputs[1017] -0.00815309*inputs[1018] +0.000419841*inputs[1019] +0.0292382*inputs[1020] +0.000419846*inputs[1021] -0.00534417*inputs[1022] +0.000419847*inputs[1023] -0.0121489*inputs[1024] +0.000419849*inputs[1025] -0.0121489*inputs[1026] +0.0159616*inputs[1027] +0.000419847*inputs[1028] -0.00765509*inputs[1029] +0.0292837*inputs[1030] +0.0164544*inputs[1031] +0.0164544*inputs[1032] -0.00765509*inputs[1033] +0.000419848*inputs[1034] +0.000419841*inputs[1035] -0.00765509*inputs[1036] -0.00777659*inputs[1037] +0.0108644*inputs[1038] +0.000419849*inputs[1039] +0.0221759*inputs[1040] +0.000419847*inputs[1041] +0.00702541*inputs[1042] +0.00702543*inputs[1043] -0.0112429*inputs[1044] -0.0112429*inputs[1045] -0.0112429*inputs[1046] -0.00777659*inputs[1047] -0.00777659*inputs[1048] +0.00884189*inputs[1049] -0.0194364*inputs[1050] +0.0147953*inputs[1051] +0.0147954*inputs[1052] +0.000419841*inputs[1053] +0.0325489*inputs[1054] +0.000419849*inputs[1055] +0.000419849*inputs[1056] +0.042151*inputs[1057] +0.000419846*inputs[1058] +0.000419849*inputs[1059] +0.000419848*inputs[1060] +0.0112443*inputs[1061] -0.00641953*inputs[1062] +0.00598612*inputs[1063] +0.0059861*inputs[1064] +0.000419841*inputs[1065] +0.000419847*inputs[1066] +0.0182621*inputs[1067] +0.0182621*inputs[1068] +0.00713847*inputs[1069] +0.0017653*inputs[1070] -0.0273023*inputs[1071] +0.0112443*inputs[1072] +0.000419847*inputs[1073] +0.000419846*inputs[1074] +0.000419845*inputs[1075] +0.000419845*inputs[1076] +0.00380432*inputs[1077] +0.00380431*inputs[1078] +0.00380431*inputs[1079] -0.00317817*inputs[1080] -0.00317816*inputs[1081] +0.000419848*inputs[1082] +0.000419847*inputs[1083] +0.00811817*inputs[1084] +0.000419845*inputs[1085] +0.000419849*inputs[1086] +0.000419845*inputs[1087] +0.000419841*inputs[1088] +0.000419849*inputs[1089] -0.00765509*inputs[1090] -0.0307782*inputs[1091] +0.000419849*inputs[1092] -0.0141318*inputs[1093] +0.000419849*inputs[1094] +0.0245778*inputs[1095] -0.0251616*inputs[1096] +0.000419849*inputs[1097] -0.0169406*inputs[1098] +0.000419849*inputs[1099] +0.0159998*inputs[1100] +0.0354655*inputs[1101] -0.0390908*inputs[1102] -0.00720967*inputs[1103] -0.00807104*inputs[1104] +0.0148782*inputs[1105] +0.000419849*inputs[1106] -0.00720967*inputs[1107] +0.000419844*inputs[1108] -0.0166775*inputs[1109] +0.000419849*inputs[1110] +0.0111868*inputs[1111] +0.000419848*inputs[1112] -0.0192068*inputs[1113] +0.00731465*inputs[1114] +0.0168033*inputs[1115] -0.0166775*inputs[1116] -0.00730029*inputs[1117] -0.0261599*inputs[1118] +0.000419842*inputs[1119] +0.000419849*inputs[1120] +0.00578252*inputs[1121] +0.0110943*inputs[1122] -0.0323296*inputs[1123] +0.000419849*inputs[1124] -0.00230939*inputs[1125] +0.00731465*inputs[1126] -0.00796051*inputs[1127] +0.000419849*inputs[1128] -0.00730029*inputs[1129] +0.000419849*inputs[1130] +0.000419849*inputs[1131] +0.000419849*inputs[1132] +0.000419845*inputs[1133] +0.000419848*inputs[1134] +0.00590287*inputs[1135] +0.000419842*inputs[1136] -0.0157251*inputs[1137] +0.0344039*inputs[1138] +0.00731464*inputs[1139] +0.000419849*inputs[1140] -0.00522816*inputs[1141] -0.00522816*inputs[1142] +0.000419844*inputs[1143] +0.00731465*inputs[1144] +0.000419849*inputs[1145] +0.000419849*inputs[1146] +0.000419848*inputs[1147] +0.000419849*inputs[1148] +0.000419849*inputs[1149] -0.0207124*inputs[1150] +0.000419849*inputs[1151] +0.000419849*inputs[1152] +0.000419846*inputs[1153] +0.00578252*inputs[1154] +0.00578252*inputs[1155] +0.000419846*inputs[1156] +0.0149177*inputs[1157] -0.0110454*inputs[1158] -0.0110454*inputs[1159] +0.0203553*inputs[1160] -0.0102215*inputs[1161] -0.0248893*inputs[1162] +0.000419849*inputs[1163] +0.000419849*inputs[1164] -0.0102215*inputs[1165] +0.0289712*inputs[1166] +0.000419849*inputs[1167] +0.000419848*inputs[1168] -0.0110454*inputs[1169] +0.00788091*inputs[1170] +0.000419849*inputs[1171] +0.0109826*inputs[1172] +0.000419847*inputs[1173] +0.00788091*inputs[1174] +0.000419849*inputs[1175] +0.000419849*inputs[1176] +0.000419848*inputs[1177] +0.0149177*inputs[1178] +0.000419848*inputs[1179] +0.000419845*inputs[1180] -0.0116246*inputs[1181] +0.000419849*inputs[1182] +0.000419849*inputs[1183] +0.000419848*inputs[1184] +0.000419849*inputs[1185] +0.000419849*inputs[1186] +0.00731464*inputs[1187] +0.00354241*inputs[1188] +0.00354241*inputs[1189] +0.00919567*inputs[1190] -0.0116246*inputs[1191] -0.0116246*inputs[1192] +0.0143947*inputs[1193] -0.0116246*inputs[1194] +0.000419847*inputs[1195] +0.000419845*inputs[1196] +0.00919567*inputs[1197] +0.0167871*inputs[1198] +0.0167871*inputs[1199] -0.0344109*inputs[1200] +0.000419847*inputs[1201] +0.000419848*inputs[1202] +0.000419848*inputs[1203] +0.000419843*inputs[1204] -0.00768222*inputs[1205] +0.011732*inputs[1206] +0.000419849*inputs[1207] +0.000419844*inputs[1208] +0.000419849*inputs[1209] +0.000419848*inputs[1210] +0.000419846*inputs[1211] -0.00227217*inputs[1212] +0.0250312*inputs[1213] -0.00227217*inputs[1214] +0.000419849*inputs[1215] +0.000419849*inputs[1216] +0.000419849*inputs[1217] -0.00768223*inputs[1218] +0.0267462*inputs[1219] +0.000419848*inputs[1220] -0.0079525*inputs[1221] -0.0079525*inputs[1222] -0.0079525*inputs[1223] -0.0079525*inputs[1224] +0.0100036*inputs[1225] +0.000419848*inputs[1226] +0.000419849*inputs[1227] -0.00329744*inputs[1228] +0.000419847*inputs[1229] +0.00842633*inputs[1230] +0.00431935*inputs[1231] +0.00431935*inputs[1232] -0.00411741*inputs[1233] -0.00411742*inputs[1234] -0.00309355*inputs[1235] +0.000378601*inputs[1236] +0.000419842*inputs[1237] +0.000378605*inputs[1238] +0.000378599*inputs[1239] -0.00309354*inputs[1240] -0.00329744*inputs[1241] -0.0118516*inputs[1242] +0.000419845*inputs[1243] -0.0185748*inputs[1244] -0.0185748*inputs[1245] +0.000419849*inputs[1246] +0.000419849*inputs[1247] -0.017925*inputs[1248] -0.017925*inputs[1249] +0.000419841*inputs[1250] +0.000419848*inputs[1251] +0.000419849*inputs[1252] +0.00185837*inputs[1253] +0.000419848*inputs[1254] +0.000419848*inputs[1255] +0.000419847*inputs[1256] +0.000419849*inputs[1257] +0.00657165*inputs[1258] +0.000419849*inputs[1259] +0.00657165*inputs[1260] +0.0304737*inputs[1261] +0.000419848*inputs[1262] +0.000419849*inputs[1263] +0.00185837*inputs[1264] +0.000419849*inputs[1265] +0.00185838*inputs[1266] +0.0102699*inputs[1267] +0.000419849*inputs[1268] -0.0171235*inputs[1269] +0.000419849*inputs[1270] +0.000419846*inputs[1271] +0.0213289*inputs[1272] +0.000419849*inputs[1273] +0.000419843*inputs[1274] +0.000419849*inputs[1275] +0.000419849*inputs[1276] +0.00499091*inputs[1277] +0.000419848*inputs[1278] -0.0300757*inputs[1279] +0.000419848*inputs[1280] -0.0134156*inputs[1281] +0.000419849*inputs[1282] +0.000419844*inputs[1283] -0.016712*inputs[1284] +0.000419843*inputs[1285] +0.00499091*inputs[1286] +0.00499091*inputs[1287] +0.00499091*inputs[1288] +0.000419848*inputs[1289] +0.0376083*inputs[1290] +0.028523*inputs[1291] +0.000419849*inputs[1292] +0.0214218*inputs[1293] +0.000419848*inputs[1294] -0.0389355*inputs[1295] -0.000637434*inputs[1296] -0.000637434*inputs[1297] +0.000419843*inputs[1298] +0.00782635*inputs[1299] +0.000419845*inputs[1300] +0.00451589*inputs[1301] -0.0122073*inputs[1302] +0.00454811*inputs[1303] +0.000419849*inputs[1304] +0.000419849*inputs[1305] +0.0316959*inputs[1306] +0.000419848*inputs[1307] +0.000419844*inputs[1308] -0.00186812*inputs[1309] +0.000419849*inputs[1310] +0.00451589*inputs[1311] +0.00451589*inputs[1312] +0.00454811*inputs[1313] +0.00451589*inputs[1314] +0.00451589*inputs[1315] +0.000419849*inputs[1316] +0.000419849*inputs[1317] -0.00339563*inputs[1318] +0.000419846*inputs[1319] +0.000419848*inputs[1320] +0.000419849*inputs[1321] -0.0192713*inputs[1322] -0.0106628*inputs[1323] -0.0106628*inputs[1324] -0.010376*inputs[1325] -0.00979802*inputs[1326] -0.00979802*inputs[1327] -0.00979802*inputs[1328] -0.00979802*inputs[1329] -0.0162914*inputs[1330] -0.0236371*inputs[1331] -0.00338547*inputs[1332] -0.00338547*inputs[1333] +0.000419849*inputs[1334] +0.000419847*inputs[1335] +0.000419849*inputs[1336] +0.0170513*inputs[1337] +0.000419845*inputs[1338] +0.00811817*inputs[1339] +0.0290937*inputs[1340] +0.000419848*inputs[1341] +0.00458066*inputs[1342] +0.00458069*inputs[1343] +0.000419848*inputs[1344] +0.000419849*inputs[1345] +0.000419846*inputs[1346] +0.000419849*inputs[1347] +0.00454811*inputs[1348] -0.00746376*inputs[1349] +0.0273521*inputs[1350] +0.0112043*inputs[1351] +0.0112042*inputs[1352] -0.00339563*inputs[1353] -0.0124246*inputs[1354] +0.00839031*inputs[1355] +0.00839031*inputs[1356] +0.00839031*inputs[1357] -0.00746376*inputs[1358] -0.00746376*inputs[1359] -0.0137532*inputs[1360] -0.00339563*inputs[1361] +0.000419844*inputs[1362] +0.00431934*inputs[1363] +0.0071385*inputs[1364] -0.0111531*inputs[1365] +0.00431935*inputs[1366] +0.000419849*inputs[1367] -0.0111531*inputs[1368] -0.0111531*inputs[1369] +0.000419849*inputs[1370] +0.00431935*inputs[1371] +0.000419844*inputs[1372] +0.00851831*inputs[1373] +0.000419845*inputs[1374] +0.0167606*inputs[1375] +0.0117072*inputs[1376] +0.00539005*inputs[1377] +0.00539003*inputs[1378] +0.00539002*inputs[1379] +0.000419849*inputs[1380] +0.0089146*inputs[1381] +0.000419848*inputs[1382] +0.0376259*inputs[1383] +0.000419849*inputs[1384] +0.000419849*inputs[1385] +0.00851831*inputs[1386] +0.00841555*inputs[1387] +0.00841555*inputs[1388] +0.00841555*inputs[1389] +0.000419849*inputs[1390] +0.000419844*inputs[1391] +0.000419844*inputs[1392] -0.00339563*inputs[1393] +0.000419848*inputs[1394] -0.0203097*inputs[1395] 
		combinations[5] = -0.0132828 -0.0559225*inputs[0] -0.0577632*inputs[1] -0.233474*inputs[2] -0.161132*inputs[3] -0.0883757*inputs[4] -0.0781006*inputs[5] -0.0454774*inputs[6] -0.00259439*inputs[7] -0.0295348*inputs[8] +0.000960978*inputs[9] -0.0024978*inputs[10] -0.10662*inputs[11] -0.0263191*inputs[12] -0.110599*inputs[13] -0.0453981*inputs[14] -0.124268*inputs[15] +0.00351459*inputs[16] -0.0108671*inputs[17] -0.104781*inputs[18] -0.00169829*inputs[19] -0.0407954*inputs[20] +0.0856745*inputs[21] -0.0291381*inputs[22] -0.0327502*inputs[23] -0.125458*inputs[24] +0.0087005*inputs[25] +0.0666357*inputs[26] +0.00970258*inputs[27] +0.00986245*inputs[28] +0.0418887*inputs[29] +0.00921583*inputs[30] +0.0737046*inputs[31] +0.0365276*inputs[32] -0.0969132*inputs[33] +0.0540668*inputs[34] -0.109167*inputs[35] -0.0700365*inputs[36] -0.0403794*inputs[37] -0.0130151*inputs[38] +0.114051*inputs[39] +0.058287*inputs[40] -0.129854*inputs[41] -0.0592039*inputs[42] -0.0423952*inputs[43] +0.0120527*inputs[44] -0.0154989*inputs[45] +0.035708*inputs[46] -0.0744155*inputs[47] +0.041355*inputs[48] +0.00449925*inputs[49] -0.103726*inputs[50] -0.0106888*inputs[51] +0.0178973*inputs[52] +0.0166595*inputs[53] -0.00646769*inputs[54] +0.0624415*inputs[55] +0.0261627*inputs[56] -0.0309247*inputs[57] +0.10978*inputs[58] +0.00503815*inputs[59] -0.0246615*inputs[60] +0.117347*inputs[61] +0.0381388*inputs[62] +0.0241444*inputs[63] +0.0420771*inputs[64] +0.0224465*inputs[65] -0.071758*inputs[66] +0.0632182*inputs[67] -0.0221332*inputs[68] +0.0109744*inputs[69] +0.0116803*inputs[70] -0.0228324*inputs[71] +0.00251289*inputs[72] -0.0274353*inputs[73] +0.087315*inputs[74] -0.00709086*inputs[75] -0.0261781*inputs[76] +0.0366617*inputs[77] +0.182008*inputs[78] -0.0242319*inputs[79] +0.00633182*inputs[80] -0.0134802*inputs[81] -0.077345*inputs[82] +0.000612384*inputs[83] -0.0143998*inputs[84] +0.0272438*inputs[85] -0.042758*inputs[86] +0.0434946*inputs[87] +0.0259287*inputs[88] -0.0128479*inputs[89] +0.0122034*inputs[90] -0.0305768*inputs[91] +0.0544192*inputs[92] +0.0666048*inputs[93] -0.0572408*inputs[94] +0.0651679*inputs[95] +0.013989*inputs[96] +0.0400605*inputs[97] +0.0100065*inputs[98] -0.00759324*inputs[99] +0.0059358*inputs[100] +0.00584004*inputs[101] -0.00906974*inputs[102] +0.0641816*inputs[103] +0.0229975*inputs[104] +2.80509e-05*inputs[105] -0.0168342*inputs[106] -0.0129398*inputs[107] -0.0159329*inputs[108] +0.00445699*inputs[109] -0.0495098*inputs[110] -0.0608376*inputs[111] -0.0248783*inputs[112] +0.00870144*inputs[113] -0.0749459*inputs[114] +0.0408083*inputs[115] +0.0530085*inputs[116] -0.00190366*inputs[117] -0.0462411*inputs[118] +0.0177829*inputs[119] +0.0166299*inputs[120] +0.00273064*inputs[121] +0.0576976*inputs[122] +0.00201057*inputs[123] +0.00887325*inputs[124] +0.0170142*inputs[125] +0.00472821*inputs[126] +0.0113903*inputs[127] +0.066128*inputs[128] -0.0249826*inputs[129] +0.00491034*inputs[130] -0.0380836*inputs[131] -0.0256644*inputs[132] -0.090037*inputs[133] +0.0179269*inputs[134] +0.0235246*inputs[135] +0.0093763*inputs[136] +0.0472615*inputs[137] -0.00720219*inputs[138] -0.00929387*inputs[139] -0.0062734*inputs[140] -0.0282377*inputs[141] -0.0206124*inputs[142] +0.10681*inputs[143] -0.0105579*inputs[144] +0.0346347*inputs[145] +0.00842731*inputs[146] -0.00831496*inputs[147] +0.00729216*inputs[148] +0.00125776*inputs[149] +0.0161701*inputs[150] -0.0630037*inputs[151] +0.0543963*inputs[152] +0.0385034*inputs[153] -0.0119393*inputs[154] +0.00159164*inputs[155] +0.0233997*inputs[156] +0.00862961*inputs[157] +0.0604161*inputs[158] -0.0154353*inputs[159] -0.0349254*inputs[160] +0.0110365*inputs[161] +0.0179551*inputs[162] +0.00519719*inputs[163] +0.0351181*inputs[164] +0.00541969*inputs[165] +0.0101802*inputs[166] +0.0365422*inputs[167] +0.024879*inputs[168] -0.0541045*inputs[169] +0.0116569*inputs[170] +0.0199317*inputs[171] +0.0306553*inputs[172] +0.019943*inputs[173] +0.00414385*inputs[174] -0.00612509*inputs[175] +0.0284372*inputs[176] -0.0119484*inputs[177] +0.0330782*inputs[178] -0.00889531*inputs[179] +0.0358763*inputs[180] -0.00379235*inputs[181] +0.0513435*inputs[182] +0.00652347*inputs[183] +0.0842165*inputs[184] -0.01876*inputs[185] +0.0379409*inputs[186] -0.0171882*inputs[187] -0.00644736*inputs[188] +0.0236872*inputs[189] -0.0737742*inputs[190] -0.0453873*inputs[191] -0.0109581*inputs[192] -0.0197021*inputs[193] +0.0041577*inputs[194] +0.00355187*inputs[195] +0.00410785*inputs[196] +0.00737083*inputs[197] -0.0246749*inputs[198] -0.0365427*inputs[199] -0.00666101*inputs[200] -0.0546667*inputs[201] -0.0227507*inputs[202] +0.0152759*inputs[203] +0.00262746*inputs[204] -0.00694672*inputs[205] -0.0331597*inputs[206] +0.002192*inputs[207] +0.0229626*inputs[208] -0.0525111*inputs[209] +0.058296*inputs[210] -0.00820114*inputs[211] +0.000941158*inputs[212] -0.00873078*inputs[213] -0.0224948*inputs[214] +0.0031234*inputs[215] +0.0145176*inputs[216] -0.00616713*inputs[217] -0.0389706*inputs[218] +0.00295684*inputs[219] +0.0209387*inputs[220] -0.0180349*inputs[221] +0.0074428*inputs[222] +0.0472491*inputs[223] -0.00925735*inputs[224] +0.0469456*inputs[225] -0.0164544*inputs[226] +0.0101779*inputs[227] +0.0273205*inputs[228] +0.0618388*inputs[229] -0.0279688*inputs[230] +0.0137345*inputs[231] -0.0010538*inputs[232] +0.00584717*inputs[233] +0.00657248*inputs[234] +0.00461159*inputs[235] +0.0232357*inputs[236] -0.00441671*inputs[237] -0.0169826*inputs[238] +0.016371*inputs[239] -0.0302401*inputs[240] -0.0207057*inputs[241] +0.00720543*inputs[242] -0.0571199*inputs[243] -0.0206661*inputs[244] +0.0480746*inputs[245] -0.0100844*inputs[246] -0.020617*inputs[247] +0.0237956*inputs[248] +0.0564078*inputs[249] +0.0139913*inputs[250] +0.0326376*inputs[251] -0.0145347*inputs[252] -0.0208395*inputs[253] -0.00754251*inputs[254] -0.033789*inputs[255] -0.0425792*inputs[256] -0.0766208*inputs[257] -0.0165109*inputs[258] -0.0369324*inputs[259] +0.0210438*inputs[260] +0.0139025*inputs[261] +0.0519376*inputs[262] -0.0320811*inputs[263] -0.00901558*inputs[264] +0.0193022*inputs[265] +0.0180132*inputs[266] +0.00596465*inputs[267] -0.0141468*inputs[268] +0.014583*inputs[269] -0.0368408*inputs[270] -0.0641056*inputs[271] +0.00474159*inputs[272] -0.000603942*inputs[273] +0.00863799*inputs[274] -0.0307988*inputs[275] +0.0369129*inputs[276] +0.0438694*inputs[277] +0.00329341*inputs[278] +0.0524559*inputs[279] -0.0823*inputs[280] +0.00423943*inputs[281] -0.005934*inputs[282] +0.00581842*inputs[283] -0.00326711*inputs[284] +0.017617*inputs[285] +0.0142489*inputs[286] -0.0657381*inputs[287] -0.00173058*inputs[288] +0.00879793*inputs[289] +0.000319813*inputs[290] +0.00459563*inputs[291] +0.00510012*inputs[292] +0.00311927*inputs[293] +0.00488587*inputs[294] -0.0163626*inputs[295] -0.0174289*inputs[296] +0.0160572*inputs[297] +0.00158446*inputs[298] +0.00967263*inputs[299] +0.00107158*inputs[300] +0.00852021*inputs[301] +0.0089062*inputs[302] -0.0166385*inputs[303] +0.0037292*inputs[304] -0.00625817*inputs[305] +0.0274921*inputs[306] +0.00724708*inputs[307] +0.00192871*inputs[308] +0.0134403*inputs[309] -0.000750215*inputs[310] +0.0169539*inputs[311] -0.0147178*inputs[312] +0.0178927*inputs[313] +0.0332024*inputs[314] -0.00335246*inputs[315] +0.0148701*inputs[316] +0.0240605*inputs[317] +0.00829798*inputs[318] +0.0319133*inputs[319] -0.0337163*inputs[320] -0.00717548*inputs[321] +0.000563793*inputs[322] -0.0068385*inputs[323] -0.0104502*inputs[324] +0.0259222*inputs[325] +0.000728291*inputs[326] +0.0104674*inputs[327] -0.0174483*inputs[328] +0.013495*inputs[329] +0.0116308*inputs[330] +0.0144001*inputs[331] -0.0108399*inputs[332] +0.000520418*inputs[333] -0.0263925*inputs[334] +0.0180156*inputs[335] +0.0170363*inputs[336] -0.0280793*inputs[337] +0.00824082*inputs[338] +0.00217436*inputs[339] -0.018243*inputs[340] +0.0321475*inputs[341] +0.0277193*inputs[342] +0.0163414*inputs[343] +0.0150102*inputs[344] +0.00569167*inputs[345] +0.0152299*inputs[346] +0.0539927*inputs[347] +0.0315592*inputs[348] +0.0144209*inputs[349] -0.019448*inputs[350] +0.000728291*inputs[351] +0.0100639*inputs[352] -0.0290992*inputs[353] +0.00154863*inputs[354] +0.00997302*inputs[355] -0.0126538*inputs[356] -0.0120132*inputs[357] -0.013262*inputs[358] -0.0235077*inputs[359] +0.00833175*inputs[360] +0.00900362*inputs[361] +0.0108298*inputs[362] -0.0183067*inputs[363] +0.00333669*inputs[364] +0.017392*inputs[365] -0.0071483*inputs[366] +0.00626785*inputs[367] +0.00879822*inputs[368] +0.0248566*inputs[369] -0.0438896*inputs[370] +0.00454628*inputs[371] +0.00236986*inputs[372] +0.065534*inputs[373] +0.0176225*inputs[374] +0.000728291*inputs[375] +0.0156294*inputs[376] -0.0378771*inputs[377] +0.041911*inputs[378] +0.0292822*inputs[379] +0.0114333*inputs[380] +0.00502001*inputs[381] -0.00382934*inputs[382] +0.017477*inputs[383] +0.0395467*inputs[384] -0.00931013*inputs[385] +0.011425*inputs[386] +0.00420138*inputs[387] -0.000482098*inputs[388] -0.0144929*inputs[389] -0.00233016*inputs[390] +0.000728291*inputs[391] -0.0128841*inputs[392] -0.023437*inputs[393] +0.000728291*inputs[394] -0.0147475*inputs[395] -0.00317044*inputs[396] +0.0449304*inputs[397] +0.0323288*inputs[398] +0.0414703*inputs[399] +0.0158418*inputs[400] +0.000728291*inputs[401] +0.00698588*inputs[402] +0.0131294*inputs[403] -0.00346945*inputs[404] -0.0499087*inputs[405] +0.00223845*inputs[406] -0.0144479*inputs[407] -0.0289367*inputs[408] -0.0227631*inputs[409] +0.0148735*inputs[410] -0.00983783*inputs[411] +0.0145352*inputs[412] -0.00317885*inputs[413] +0.0096318*inputs[414] -0.0232*inputs[415] +0.0397466*inputs[416] -0.0260481*inputs[417] +0.0250007*inputs[418] -0.00585518*inputs[419] +0.0117939*inputs[420] +0.0448266*inputs[421] +0.00617208*inputs[422] -0.00727039*inputs[423] +0.00284244*inputs[424] +0.000728291*inputs[425] -0.00470984*inputs[426] +0.0156102*inputs[427] -0.00398772*inputs[428] +0.000117773*inputs[429] +0.000728291*inputs[430] +0.0226103*inputs[431] -0.00917753*inputs[432] +0.00311402*inputs[433] -0.0151604*inputs[434] +0.00223315*inputs[435] -0.0165555*inputs[436] -0.0124596*inputs[437] +0.00535513*inputs[438] -0.0184415*inputs[439] +0.020089*inputs[440] +0.0297608*inputs[441] +0.00625029*inputs[442] +0.0185238*inputs[443] +0.000728291*inputs[444] +0.0182892*inputs[445] +0.00527336*inputs[446] +0.0232223*inputs[447] +0.000110213*inputs[448] +0.000594356*inputs[449] -0.0198581*inputs[450] -0.0351072*inputs[451] +0.00318232*inputs[452] +0.0326125*inputs[453] -0.00501678*inputs[454] +0.0269868*inputs[455] +0.0355922*inputs[456] +0.0138099*inputs[457] -0.0571022*inputs[458] +0.0499965*inputs[459] +0.0293324*inputs[460] -0.00416239*inputs[461] +0.00349234*inputs[462] +0.0206556*inputs[463] +0.0356023*inputs[464] +0.0543935*inputs[465] -0.00030343*inputs[466] +0.0104841*inputs[467] +0.0132212*inputs[468] +0.00589838*inputs[469] +0.0161497*inputs[470] +0.0215951*inputs[471] -0.0107459*inputs[472] +0.00887304*inputs[473] -0.0180983*inputs[474] -0.0180983*inputs[475] +0.01342*inputs[476] +0.0253825*inputs[477] -0.0101026*inputs[478] +0.0107663*inputs[479] -0.00517465*inputs[480] -0.0398057*inputs[481] +0.0247504*inputs[482] +0.0210263*inputs[483] +0.000594353*inputs[484] +0.00821277*inputs[485] +0.0245393*inputs[486] +0.0220912*inputs[487] +0.0105213*inputs[488] +0.0173814*inputs[489] -0.00807642*inputs[490] +0.0238357*inputs[491] +0.0115931*inputs[492] +0.0310495*inputs[493] +0.010768*inputs[494] +0.0074519*inputs[495] +0.00745188*inputs[496] +0.0112004*inputs[497] -0.0687212*inputs[498] +0.00625214*inputs[499] +0.0210263*inputs[500] +0.00887305*inputs[501] +0.000594355*inputs[502] +0.000594353*inputs[503] +0.0269099*inputs[504] -0.0378035*inputs[505] -0.0198595*inputs[506] -0.0143762*inputs[507] +0.000594355*inputs[508] +0.0038564*inputs[509] -0.00199486*inputs[510] +0.0515843*inputs[511] +0.0180299*inputs[512] -0.00606262*inputs[513] +0.000594352*inputs[514] -0.0287683*inputs[515] +0.000594354*inputs[516] +0.0123474*inputs[517] +0.0138099*inputs[518] +0.000594352*inputs[519] -0.00199486*inputs[520] -0.00626235*inputs[521] +0.0181029*inputs[522] -0.00238407*inputs[523] +0.00556922*inputs[524] +0.00556921*inputs[525] -0.00720153*inputs[526] +0.0144896*inputs[527] +0.000594356*inputs[528] -0.00295809*inputs[529] +0.00848933*inputs[530] -0.0206198*inputs[531] -0.0107879*inputs[532] -0.0109121*inputs[533] +0.0170404*inputs[534] +0.0256395*inputs[535] +0.0144104*inputs[536] +0.0256186*inputs[537] -0.0109121*inputs[538] -0.0116037*inputs[539] +0.000594354*inputs[540] +0.0118229*inputs[541] +0.013197*inputs[542] -0.00218299*inputs[543] +0.000594355*inputs[544] +0.00848933*inputs[545] +0.00141879*inputs[546] -0.00581936*inputs[547] -0.0202118*inputs[548] -0.00600863*inputs[549] -0.00269713*inputs[550] -0.0136699*inputs[551] +0.0038096*inputs[552] +0.0151634*inputs[553] -0.002616*inputs[554] +0.000594354*inputs[555] -0.00759671*inputs[556] -0.00407458*inputs[557] -0.0129883*inputs[558] +0.040672*inputs[559] -0.00210483*inputs[560] -0.00591237*inputs[561] -0.00961582*inputs[562] -0.0362112*inputs[563] -0.0474269*inputs[564] +0.000594356*inputs[565] +0.00660455*inputs[566] +0.0109574*inputs[567] +0.000594351*inputs[568] -0.0152146*inputs[569] -0.00937855*inputs[570] +0.000594353*inputs[571] +0.0122148*inputs[572] +0.0340011*inputs[573] +0.000594356*inputs[574] -0.0265443*inputs[575] +0.000594352*inputs[576] +0.0397274*inputs[577] -0.0136753*inputs[578] +0.0250013*inputs[579] +0.00494724*inputs[580] +0.00151205*inputs[581] +0.000594356*inputs[582] +0.00848935*inputs[583] +0.00583402*inputs[584] +0.0204849*inputs[585] -0.0245248*inputs[586] +0.000594356*inputs[587] -0.0292432*inputs[588] +0.000594353*inputs[589] +0.00859923*inputs[590] +0.000420064*inputs[591] +0.000594355*inputs[592] +0.0682101*inputs[593] +0.000928152*inputs[594] -0.0276361*inputs[595] +0.0045339*inputs[596] +0.0317041*inputs[597] +0.00509957*inputs[598] +0.00919793*inputs[599] -0.0198034*inputs[600] -0.0359404*inputs[601] +0.0120315*inputs[602] +0.011621*inputs[603] +0.0183062*inputs[604] +0.000594352*inputs[605] -0.00478693*inputs[606] +0.00509956*inputs[607] -0.00839566*inputs[608] -0.0180709*inputs[609] +0.000594355*inputs[610] -0.0118222*inputs[611] -0.0319608*inputs[612] +0.00821277*inputs[613] +0.000420069*inputs[614] +0.00702969*inputs[615] -0.00188334*inputs[616] +0.0377205*inputs[617] +0.014134*inputs[618] +0.00151974*inputs[619] -0.0221434*inputs[620] +0.00432247*inputs[621] +0.0130712*inputs[622] -0.0221434*inputs[623] +0.000894152*inputs[624] -0.0125478*inputs[625] +0.0196551*inputs[626] +0.000594356*inputs[627] +0.000594356*inputs[628] +0.000594352*inputs[629] +0.0176946*inputs[630] +0.0173386*inputs[631] -0.00693613*inputs[632] +0.0357892*inputs[633] +0.00969463*inputs[634] +0.00556921*inputs[635] +0.000594356*inputs[636] -0.010832*inputs[637] +0.0223988*inputs[638] +0.000594355*inputs[639] +0.0366958*inputs[640] -0.0151246*inputs[641] -0.00339727*inputs[642] -0.0258307*inputs[643] +0.0366395*inputs[644] +0.0176946*inputs[645] +0.0568276*inputs[646] +0.0120576*inputs[647] +0.000420063*inputs[648] -0.0089133*inputs[649] -0.00273135*inputs[650] -0.0287388*inputs[651] +0.00828459*inputs[652] -0.0089133*inputs[653] +0.00828457*inputs[654] +0.00260127*inputs[655] -0.0089133*inputs[656] +0.000420062*inputs[657] +0.000420064*inputs[658] -0.00273135*inputs[659] +0.000420064*inputs[660] +0.000420067*inputs[661] +0.00407809*inputs[662] +0.000420067*inputs[663] -0.00628836*inputs[664] -0.00325332*inputs[665] +0.0382607*inputs[666] +0.0233764*inputs[667] +0.000420068*inputs[668] -0.0243436*inputs[669] +0.00443225*inputs[670] +0.0243979*inputs[671] +0.000420069*inputs[672] +0.00407809*inputs[673] -0.00899931*inputs[674] -0.0138409*inputs[675] -0.00325332*inputs[676] +0.000420069*inputs[677] +0.000420067*inputs[678] +0.000420068*inputs[679] -0.00325332*inputs[680] +0.00260127*inputs[681] +0.000420063*inputs[682] -0.000391603*inputs[683] +0.000420069*inputs[684] +0.0322823*inputs[685] -0.0121398*inputs[686] +0.000420067*inputs[687] -0.0121398*inputs[688] -0.0121398*inputs[689] +0.000420068*inputs[690] +0.0221108*inputs[691] +0.000420069*inputs[692] +0.000420069*inputs[693] +0.00443227*inputs[694] +0.00496486*inputs[695] +0.00496486*inputs[696] +0.000420067*inputs[697] -0.0311192*inputs[698] -0.0089993*inputs[699] +0.00496486*inputs[700] +0.00496486*inputs[701] -0.0149978*inputs[702] +0.000420069*inputs[703] +0.000420069*inputs[704] +0.000420064*inputs[705] +0.000420069*inputs[706] +0.000420069*inputs[707] -0.0239864*inputs[708] +0.000420069*inputs[709] +0.000420069*inputs[710] +0.000420069*inputs[711] +0.000420063*inputs[712] -0.0149978*inputs[713] -0.0149978*inputs[714] -0.0149978*inputs[715] -0.00778452*inputs[716] +0.000420069*inputs[717] -0.0208075*inputs[718] -0.0208075*inputs[719] +0.000420069*inputs[720] +0.000420069*inputs[721] +0.000420069*inputs[722] +0.000420068*inputs[723] +0.000420067*inputs[724] +0.00435381*inputs[725] -0.0061747*inputs[726] -0.0207301*inputs[727] +0.014887*inputs[728] +0.014887*inputs[729] +0.014887*inputs[730] -0.0207301*inputs[731] -0.0109714*inputs[732] +0.000420069*inputs[733] +0.000420069*inputs[734] +0.000420069*inputs[735] -0.00460133*inputs[736] -0.00460133*inputs[737] -0.00460134*inputs[738] -0.00460133*inputs[739] +0.00713983*inputs[740] -0.0207301*inputs[741] +0.0229587*inputs[742] +0.000420064*inputs[743] +0.000420069*inputs[744] +0.000420069*inputs[745] +0.00476359*inputs[746] +0.000420068*inputs[747] -0.0106802*inputs[748] +0.000420061*inputs[749] +0.000420069*inputs[750] -0.0234084*inputs[751] -0.0234084*inputs[752] +0.000420061*inputs[753] +0.000420067*inputs[754] +0.000420068*inputs[755] -0.0115556*inputs[756] +0.000420069*inputs[757] -0.0374657*inputs[758] +0.0047636*inputs[759] +0.00476359*inputs[760] +0.0135764*inputs[761] +0.0135764*inputs[762] +0.0135764*inputs[763] -0.0109714*inputs[764] +0.000420069*inputs[765] +0.000420068*inputs[766] +0.000420068*inputs[767] +0.000420069*inputs[768] +0.000420069*inputs[769] +0.000920888*inputs[770] +0.000420069*inputs[771] -0.00908161*inputs[772] +0.0210135*inputs[773] -0.0090816*inputs[774] -0.00927194*inputs[775] +0.00503099*inputs[776] -0.0090816*inputs[777] +0.00503099*inputs[778] +0.00503099*inputs[779] -0.000391604*inputs[780] -0.00793618*inputs[781] +0.0347325*inputs[782] +0.000920888*inputs[783] +0.000420069*inputs[784] -0.0265033*inputs[785] +0.004832*inputs[786] +0.00483202*inputs[787] +0.004832*inputs[788] +0.004832*inputs[789] +0.0160684*inputs[790] -0.00773447*inputs[791] +0.000420069*inputs[792] +0.000420069*inputs[793] +0.0188531*inputs[794] +0.000420069*inputs[795] +0.0121221*inputs[796] +0.0121221*inputs[797] +0.000420069*inputs[798] +0.000420069*inputs[799] +0.000420068*inputs[800] +0.0141509*inputs[801] +0.0141509*inputs[802] +0.000420069*inputs[803] +0.00443228*inputs[804] +0.000420069*inputs[805] +0.0191004*inputs[806] -0.0128413*inputs[807] -0.0332193*inputs[808] +0.000420063*inputs[809] -0.0128413*inputs[810] +0.007452*inputs[811] +0.007452*inputs[812] +0.007452*inputs[813] +0.00291841*inputs[814] +0.00291841*inputs[815] -0.0090816*inputs[816] +0.000420068*inputs[817] -0.0095891*inputs[818] +0.000420065*inputs[819] -0.00898961*inputs[820] -0.00898961*inputs[821] -0.0284916*inputs[822] +0.000420068*inputs[823] +0.000420063*inputs[824] -0.0327365*inputs[825] +0.0256395*inputs[826] +0.0256395*inputs[827] +0.000420069*inputs[828] +0.000420061*inputs[829] +0.000420068*inputs[830] +0.000420069*inputs[831] +0.000420068*inputs[832] -0.0140036*inputs[833] -0.0140036*inputs[834] -0.0140036*inputs[835] +0.000420069*inputs[836] +0.000420063*inputs[837] +0.000420069*inputs[838] +0.00192694*inputs[839] +0.00192694*inputs[840] +0.00192695*inputs[841] -0.0132991*inputs[842] +0.000420069*inputs[843] +0.000420068*inputs[844] -0.0293017*inputs[845] +0.0261704*inputs[846] +0.00986824*inputs[847] +0.000420069*inputs[848] -0.0116037*inputs[849] +0.0393189*inputs[850] -0.00718641*inputs[851] -0.00718641*inputs[852] -0.00718641*inputs[853] +0.000420069*inputs[854] -0.019546*inputs[855] +0.000420061*inputs[856] +0.000420066*inputs[857] +0.000420068*inputs[858] +0.00368393*inputs[859] +0.00368389*inputs[860] +0.000420069*inputs[861] +0.0114882*inputs[862] +0.00575518*inputs[863] +0.0114882*inputs[864] +0.000420064*inputs[865] +0.0279005*inputs[866] +0.00328176*inputs[867] +0.00328175*inputs[868] -0.00779635*inputs[869] +0.000420068*inputs[870] +0.000420063*inputs[871] +0.000420069*inputs[872] +0.039063*inputs[873] +0.000420069*inputs[874] +0.000420068*inputs[875] -0.0132089*inputs[876] -0.0132089*inputs[877] +0.020061*inputs[878] +0.0156839*inputs[879] +0.0226748*inputs[880] +0.0226748*inputs[881] +0.000420069*inputs[882] +0.0294862*inputs[883] +0.0294862*inputs[884] +0.021449*inputs[885] +0.0230871*inputs[886] +0.000420069*inputs[887] +0.000420062*inputs[888] +0.000420069*inputs[889] +0.000420065*inputs[890] -0.0174744*inputs[891] +0.000420064*inputs[892] +0.000420066*inputs[893] -0.0158443*inputs[894] -0.00522845*inputs[895] +0.000420064*inputs[896] -0.00522846*inputs[897] +0.000420069*inputs[898] +0.000420062*inputs[899] +0.000420065*inputs[900] -0.0375889*inputs[901] -0.0374657*inputs[902] +0.000420069*inputs[903] -0.0116718*inputs[904] -0.00572709*inputs[905] +0.000420069*inputs[906] +0.000420069*inputs[907] -0.0237876*inputs[908] -0.0237876*inputs[909] +0.000420069*inputs[910] +0.000420068*inputs[911] +0.000420062*inputs[912] +0.000420063*inputs[913] +0.000420069*inputs[914] +0.000420069*inputs[915] -0.0176627*inputs[916] -0.0176627*inputs[917] -0.0176627*inputs[918] +0.000420066*inputs[919] +0.000420069*inputs[920] +0.000420069*inputs[921] +0.0178771*inputs[922] +0.0178771*inputs[923] -0.00522212*inputs[924] -0.00522212*inputs[925] -0.00522212*inputs[926] +0.000420069*inputs[927] +0.000420069*inputs[928] +0.000420069*inputs[929] -0.00864576*inputs[930] -0.00864576*inputs[931] -0.00864576*inputs[932] -0.00380368*inputs[933] -0.00380368*inputs[934] +0.000420069*inputs[935] +0.0205801*inputs[936] +0.0205801*inputs[937] +0.000420066*inputs[938] +0.000420064*inputs[939] -0.00793618*inputs[940] -0.0240711*inputs[941] -0.0158486*inputs[942] -0.0309*inputs[943] -0.00877722*inputs[944] -0.00877722*inputs[945] +0.000420067*inputs[946] +0.000420069*inputs[947] +0.0022787*inputs[948] +0.0022787*inputs[949] -0.0289896*inputs[950] +0.0270382*inputs[951] +0.000420063*inputs[952] +0.000420069*inputs[953] +0.000420068*inputs[954] +0.000420069*inputs[955] +0.000420069*inputs[956] +0.000420066*inputs[957] +0.000420067*inputs[958] +0.000420069*inputs[959] +0.000420068*inputs[960] +0.000420067*inputs[961] +0.000420061*inputs[962] +0.000420069*inputs[963] -0.0221955*inputs[964] -0.0187241*inputs[965] -0.0278316*inputs[966] -0.0184988*inputs[967] +0.000420068*inputs[968] +0.000420064*inputs[969] -0.00774011*inputs[970] -0.00774011*inputs[971] +0.027195*inputs[972] +0.0271949*inputs[973] +0.000420068*inputs[974] -0.0221955*inputs[975] -0.00350574*inputs[976] +0.0269534*inputs[977] -0.00350574*inputs[978] +0.0266903*inputs[979] +0.0271127*inputs[980] +0.0110175*inputs[981] +0.000420063*inputs[982] +0.00920282*inputs[983] +0.00920282*inputs[984] -0.0089993*inputs[985] +0.00337411*inputs[986] +0.0115797*inputs[987] +0.0254833*inputs[988] +0.000420069*inputs[989] +0.000420069*inputs[990] +0.00824353*inputs[991] +0.00824353*inputs[992] +0.00824352*inputs[993] -0.0173206*inputs[994] +0.000420069*inputs[995] +0.000420067*inputs[996] +0.0138733*inputs[997] +0.0174504*inputs[998] +0.00688405*inputs[999] +0.000420069*inputs[1000] -0.0243745*inputs[1001] +0.000420063*inputs[1002] +0.0106513*inputs[1003] +0.0106513*inputs[1004] +0.000420066*inputs[1005] +0.000420065*inputs[1006] +0.000420068*inputs[1007] +0.000420066*inputs[1008] +0.000420069*inputs[1009] +0.0289016*inputs[1010] -0.0388047*inputs[1011] +0.0269534*inputs[1012] +0.000420069*inputs[1013] -0.0121542*inputs[1014] +0.000420069*inputs[1015] +0.000420069*inputs[1016] -0.00815408*inputs[1017] -0.00815408*inputs[1018] +0.000420068*inputs[1019] +0.0292535*inputs[1020] +0.000420069*inputs[1021] -0.00534321*inputs[1022] +0.000420069*inputs[1023] -0.0121542*inputs[1024] +0.000420069*inputs[1025] -0.0121542*inputs[1026] +0.0159668*inputs[1027] +0.000420069*inputs[1028] -0.00765759*inputs[1029] +0.0293007*inputs[1030] +0.016461*inputs[1031] +0.016461*inputs[1032] -0.00765759*inputs[1033] +0.000420067*inputs[1034] +0.000420069*inputs[1035] -0.00765759*inputs[1036] -0.00777997*inputs[1037] +0.0108669*inputs[1038] +0.000420069*inputs[1039] +0.0221826*inputs[1040] +0.000420063*inputs[1041] +0.00702646*inputs[1042] +0.00702647*inputs[1043] -0.0112461*inputs[1044] -0.0112461*inputs[1045] -0.0112461*inputs[1046] -0.00777997*inputs[1047] -0.00777998*inputs[1048] +0.00884114*inputs[1049] -0.0194425*inputs[1050] +0.0148006*inputs[1051] +0.0148006*inputs[1052] +0.000420069*inputs[1053] +0.0325608*inputs[1054] +0.000420067*inputs[1055] +0.000420069*inputs[1056] +0.0421698*inputs[1057] +0.000420069*inputs[1058] +0.000420069*inputs[1059] +0.000420069*inputs[1060] +0.0112477*inputs[1061] -0.0064198*inputs[1062] +0.00598867*inputs[1063] +0.00598869*inputs[1064] +0.000420069*inputs[1065] +0.000420069*inputs[1066] +0.0182681*inputs[1067] +0.0182681*inputs[1068] +0.00713982*inputs[1069] +0.00176459*inputs[1070] -0.0273157*inputs[1071] +0.0112477*inputs[1072] +0.000420069*inputs[1073] +0.000420067*inputs[1074] +0.000420068*inputs[1075] +0.000420069*inputs[1076] +0.00380341*inputs[1077] +0.00380336*inputs[1078] +0.0038034*inputs[1079] -0.00317736*inputs[1080] -0.00317736*inputs[1081] +0.000420069*inputs[1082] +0.000420069*inputs[1083] +0.00812005*inputs[1084] +0.000420069*inputs[1085] +0.000420069*inputs[1086] +0.000420068*inputs[1087] +0.000420064*inputs[1088] +0.000420066*inputs[1089] -0.00765759*inputs[1090] -0.0307941*inputs[1091] +0.000420069*inputs[1092] -0.0141318*inputs[1093] +0.000420069*inputs[1094] +0.0245914*inputs[1095] -0.0251721*inputs[1096] +0.000420069*inputs[1097] -0.0169496*inputs[1098] +0.000420065*inputs[1099] +0.0160063*inputs[1100] +0.0354852*inputs[1101] -0.03911*inputs[1102] -0.00721157*inputs[1103] -0.00807183*inputs[1104] +0.0148817*inputs[1105] +0.000420069*inputs[1106] -0.00721158*inputs[1107] +0.000420064*inputs[1108] -0.0166853*inputs[1109] +0.000420066*inputs[1110] +0.0111888*inputs[1111] +0.000420068*inputs[1112] -0.0192137*inputs[1113] +0.00731681*inputs[1114] +0.0168073*inputs[1115] -0.0166853*inputs[1116] -0.00730089*inputs[1117] -0.0261721*inputs[1118] +0.000420063*inputs[1119] +0.000420069*inputs[1120] +0.00578481*inputs[1121] +0.0110956*inputs[1122] -0.0323439*inputs[1123] +0.000420069*inputs[1124] -0.00230735*inputs[1125] +0.00731681*inputs[1126] -0.00796121*inputs[1127] +0.000420068*inputs[1128] -0.00730089*inputs[1129] +0.000420063*inputs[1130] +0.000420068*inputs[1131] +0.000420069*inputs[1132] +0.000420068*inputs[1133] +0.000420068*inputs[1134] +0.00590318*inputs[1135] +0.000420066*inputs[1136] -0.0157311*inputs[1137] +0.0344179*inputs[1138] +0.00731681*inputs[1139] +0.000420069*inputs[1140] -0.0052275*inputs[1141] -0.0052275*inputs[1142] +0.000420069*inputs[1143] +0.00731681*inputs[1144] +0.000420067*inputs[1145] +0.000420064*inputs[1146] +0.000420066*inputs[1147] +0.000420069*inputs[1148] +0.000420069*inputs[1149] -0.0207197*inputs[1150] +0.000420068*inputs[1151] +0.000420069*inputs[1152] +0.000420069*inputs[1153] +0.00578481*inputs[1154] +0.00578481*inputs[1155] +0.000420066*inputs[1156] +0.0149239*inputs[1157] -0.0110517*inputs[1158] -0.0110517*inputs[1159] +0.0203649*inputs[1160] -0.0102243*inputs[1161] -0.0249014*inputs[1162] +0.000420069*inputs[1163] +0.000420068*inputs[1164] -0.0102243*inputs[1165] +0.028988*inputs[1166] +0.000420066*inputs[1167] +0.000420069*inputs[1168] -0.0110517*inputs[1169] +0.00788206*inputs[1170] +0.000420065*inputs[1171] +0.0109849*inputs[1172] +0.000420068*inputs[1173] +0.00788204*inputs[1174] +0.000420069*inputs[1175] +0.000420069*inputs[1176] +0.000420068*inputs[1177] +0.0149239*inputs[1178] +0.000420067*inputs[1179] +0.000420068*inputs[1180] -0.0116296*inputs[1181] +0.000420069*inputs[1182] +0.000420061*inputs[1183] +0.000420069*inputs[1184] +0.000420069*inputs[1185] +0.000420069*inputs[1186] +0.00731681*inputs[1187] +0.00354027*inputs[1188] +0.00354027*inputs[1189] +0.00919793*inputs[1190] -0.0116296*inputs[1191] -0.0116296*inputs[1192] +0.0143993*inputs[1193] -0.0116296*inputs[1194] +0.000420069*inputs[1195] +0.000420069*inputs[1196] +0.00919793*inputs[1197] +0.0167918*inputs[1198] +0.0167918*inputs[1199] -0.0344279*inputs[1200] +0.000420069*inputs[1201] +0.000420066*inputs[1202] +0.000420068*inputs[1203] +0.000420069*inputs[1204] -0.00768491*inputs[1205] +0.011735*inputs[1206] +0.000420067*inputs[1207] +0.000420068*inputs[1208] +0.000420067*inputs[1209] +0.000420069*inputs[1210] +0.000420064*inputs[1211] -0.0022714*inputs[1212] +0.0250393*inputs[1213] -0.0022714*inputs[1214] +0.000420069*inputs[1215] +0.000420069*inputs[1216] +0.000420069*inputs[1217] -0.00768491*inputs[1218] +0.0267582*inputs[1219] +0.000420069*inputs[1220] -0.00795503*inputs[1221] -0.00795503*inputs[1222] -0.00795503*inputs[1223] -0.00795503*inputs[1224] +0.0100052*inputs[1225] +0.000420067*inputs[1226] +0.000420069*inputs[1227] -0.0032985*inputs[1228] +0.000420069*inputs[1229] +0.00842843*inputs[1230] +0.00431952*inputs[1231] +0.00431952*inputs[1232] -0.00411779*inputs[1233] -0.00411779*inputs[1234] -0.00309248*inputs[1235] +0.000378957*inputs[1236] +0.000420068*inputs[1237] +0.000378957*inputs[1238] +0.000378958*inputs[1239] -0.00309247*inputs[1240] -0.0032985*inputs[1241] -0.0118534*inputs[1242] +0.000420069*inputs[1243] -0.0185838*inputs[1244] -0.0185838*inputs[1245] +0.000420069*inputs[1246] +0.000420066*inputs[1247] -0.0179332*inputs[1248] -0.0179332*inputs[1249] +0.000420064*inputs[1250] +0.000420069*inputs[1251] +0.000420069*inputs[1252] +0.00185783*inputs[1253] +0.000420069*inputs[1254] +0.000420069*inputs[1255] +0.000420069*inputs[1256] +0.000420069*inputs[1257] +0.00657286*inputs[1258] +0.000420062*inputs[1259] +0.00657287*inputs[1260] +0.0304863*inputs[1261] +0.000420069*inputs[1262] +0.000420066*inputs[1263] +0.00185783*inputs[1264] +0.000420063*inputs[1265] +0.00185783*inputs[1266] +0.0102741*inputs[1267] +0.000420068*inputs[1268] -0.0171308*inputs[1269] +0.000420069*inputs[1270] +0.000420069*inputs[1271] +0.0213349*inputs[1272] +0.000420069*inputs[1273] +0.000420066*inputs[1274] +0.000420064*inputs[1275] +0.000420064*inputs[1276] +0.00499186*inputs[1277] +0.000420067*inputs[1278] -0.0300914*inputs[1279] +0.000420069*inputs[1280] -0.0134202*inputs[1281] +0.000420063*inputs[1282] +0.000420068*inputs[1283] -0.0167202*inputs[1284] +0.000420069*inputs[1285] +0.00499186*inputs[1286] +0.00499186*inputs[1287] +0.00499185*inputs[1288] +0.000420069*inputs[1289] +0.0376284*inputs[1290] +0.0285354*inputs[1291] +0.000420064*inputs[1292] +0.0214282*inputs[1293] +0.000420069*inputs[1294] -0.0389557*inputs[1295] -0.000636323*inputs[1296] -0.000636323*inputs[1297] +0.000420069*inputs[1298] +0.00782641*inputs[1299] +0.000420069*inputs[1300] +0.00451641*inputs[1301] -0.0122113*inputs[1302] +0.00454814*inputs[1303] +0.000420069*inputs[1304] +0.000420069*inputs[1305] +0.0317089*inputs[1306] +0.000420069*inputs[1307] +0.000420068*inputs[1308] -0.0018668*inputs[1309] +0.000420069*inputs[1310] +0.00451641*inputs[1311] +0.00451643*inputs[1312] +0.00454815*inputs[1313] +0.0045164*inputs[1314] +0.00451642*inputs[1315] +0.000420069*inputs[1316] +0.000420068*inputs[1317] -0.00339524*inputs[1318] +0.000420069*inputs[1319] +0.000420068*inputs[1320] +0.000420069*inputs[1321] -0.0192804*inputs[1322] -0.010666*inputs[1323] -0.010666*inputs[1324] -0.0103781*inputs[1325] -0.00980193*inputs[1326] -0.00980193*inputs[1327] -0.00980193*inputs[1328] -0.00980193*inputs[1329] -0.0162977*inputs[1330] -0.0236477*inputs[1331] -0.00338408*inputs[1332] -0.00338408*inputs[1333] +0.000420069*inputs[1334] +0.000420068*inputs[1335] +0.000420069*inputs[1336] +0.0170557*inputs[1337] +0.000420069*inputs[1338] +0.00812007*inputs[1339] +0.0291033*inputs[1340] +0.000420069*inputs[1341] +0.00458051*inputs[1342] +0.00458052*inputs[1343] +0.000420068*inputs[1344] +0.000420069*inputs[1345] +0.000420069*inputs[1346] +0.000420069*inputs[1347] +0.00454812*inputs[1348] -0.00746607*inputs[1349] +0.0273625*inputs[1350] +0.0112098*inputs[1351] +0.0112098*inputs[1352] -0.00339524*inputs[1353] -0.0124271*inputs[1354] +0.0083925*inputs[1355] +0.0083925*inputs[1356] +0.00839249*inputs[1357] -0.00746607*inputs[1358] -0.00746607*inputs[1359] -0.0137577*inputs[1360] -0.00339524*inputs[1361] +0.000420068*inputs[1362] +0.00431957*inputs[1363] +0.00713982*inputs[1364] -0.011158*inputs[1365] +0.00431954*inputs[1366] +0.000420069*inputs[1367] -0.011158*inputs[1368] -0.011158*inputs[1369] +0.000420069*inputs[1370] +0.00431955*inputs[1371] +0.000420069*inputs[1372] +0.00851945*inputs[1373] +0.000420069*inputs[1374] +0.0167652*inputs[1375] +0.011709*inputs[1376] +0.00539066*inputs[1377] +0.00539067*inputs[1378] +0.00539066*inputs[1379] +0.000420067*inputs[1380] +0.00891556*inputs[1381] +0.000420069*inputs[1382] +0.0376407*inputs[1383] +0.000420067*inputs[1384] +0.000420068*inputs[1385] +0.00851945*inputs[1386] +0.00841739*inputs[1387] +0.00841739*inputs[1388] +0.00841739*inputs[1389] +0.000420068*inputs[1390] +0.000420067*inputs[1391] +0.000420069*inputs[1392] -0.00339524*inputs[1393] +0.000420068*inputs[1394] -0.0203172*inputs[1395] 
		
		activations = [None] * 6

		activations[0] = np.tanh(combinations[0])
		activations[1] = np.tanh(combinations[1])
		activations[2] = np.tanh(combinations[2])
		activations[3] = np.tanh(combinations[3])
		activations[4] = np.tanh(combinations[4])
		activations[5] = np.tanh(combinations[5])

		return activations;


	def probabilistic_layer(self, inputs):

		combinations = [None] * 1

		combinations[0] = -0.0411191 -1.17193*inputs[0] -1.20362*inputs[1] +1.2195*inputs[2] -1.18311*inputs[3] +1.19279*inputs[4] +1.19414*inputs[5] 
		
		activations = [None] * 1

		activations[0] = 1.0/(1.0 + np.exp(-combinations[0]));

		return activations;


	def calculate_output(self, inputs):

		output_scaling_layer = self.scaling_layer(inputs)

		output_perceptron_layer_1 = self.perceptron_layer_1(output_scaling_layer)

		output_probabilistic_layer = self.probabilistic_layer(output_perceptron_layer_1)

		return output_probabilistic_layer


	def calculate_batch_output(self, input_batch):

		output = []

		for i in range(input_batch.shape[0]):

			inputs = list(input_batch[i])

			output_scaling_layer = self.scaling_layer(inputs)

			output_perceptron_layer_1 = self.perceptron_layer_1(output_scaling_layer)

			output_probabilistic_layer = self.probabilistic_layer(output_perceptron_layer_1)

			output = np.append(output,output_probabilistic_layer, axis=0)

		return output
