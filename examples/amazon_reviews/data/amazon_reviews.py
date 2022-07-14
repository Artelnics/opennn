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
	15 )price
	16 )like
	17 )get
	18 )time
	19 )excel
	20 )don
	21 )us
	22 )ve
	23 )recommend
	24 )realli
	25 )look
	26 )problem
	27 )comfort
	28 )call
	29 )can
	30 )charg
	31 )servic
	32 )make
	33 )buy
	34 )love
	35 )fit
	36 )nice
	37 )best
	38 )charger
	39 )also
	40 )disappoint
	41 )purchas
	42 )just
	43 )item
	44 )new
	45 )better
	46 )bluetooth
	47 )ever
	48 )money
	49 )first
	50 )easi
	51 )car
	52 )even
	53 )year
	54 )tri
	55 )bought
	56 )wast
	57 )now
	58 )recept
	59 )doesn
	60 )will
	61 )plug
	62 )last
	63 )drop
	64 )happi
	65 )thing
	66 )poor
	67 )bad
	68 )devic
	69 )still
	70 )two
	71 )high
	72 )motorola
	73 )cell
	74 )made
	75 )worst
	76 )camera
	77 )far
	78 )design
	79 )life
	80 )day
	81 )enough
	82 )piec
	83 )long
	84 )fine
	85 )got
	86 )hear
	87 )clear
	88 )volum
	89 )right
	90 )wear
	91 )go
	92 )much
	93 )screen
	94 )pictur
	95 )impress
	96 )month
	97 )say
	98 )hold
	99 )turn
	100 )pretti
	101 )lot
	102 )week
	103 )think
	104 )need
	105 )button
	106 )couldn
	107 )want
	108 )peopl
	109 )terribl
	110 )light
	111 )expect
	112 )everyth
	113 )amazon
	114 )verizon
	115 )low
	116 )receiv
	117 )cheap
	118 )unit
	119 )custom
	120 )take
	121 )talk
	122 )order
	123 )cool
	124 )return
	125 )howev
	126 )replac
	127 )end
	128 )connect
	129 )littl
	130 )voic
	131 )pair
	132 )featur
	133 )hand
	134 )feel
	135 )never
	136 )back
	137 )broke
	138 )found
	139 )junk
	140 )horribl
	141 )sever
	142 )jabra
	143 )without
	144 )quit
	145 )small
	146 )seem
	147 )color
	148 )way
	149 )nokia
	150 )stay
	151 )minut
	152 )hour
	153 )didn
	154 )loud
	155 )break
	156 )compani
	157 )internet
	158 )sinc
	159 )start
	160 )ship
	161 )quick
	162 )keep
	163 )real
	164 )cabl
	165 )place
	166 )went
	167 )softwar
	168 )find
	169 )pleas
	170 )signal
	171 )big
	172 )useless
	173 )earpiec
	174 )audio
	175 )complet
	176 )perfect
	177 )help
	178 )clip
	179 )anyon
	180 )simpl
	181 )difficult
	182 )know
	183 )everi
	184 )around
	185 )arriv
	186 )perform
	187 )contact
	188 )less
	189 )side
	190 )send
	191 )put
	192 )hard
	193 )protect
	194 )headphon
	195 )sturdi
	196 )samsung
	197 )bar
	198 )crap
	199 )origin
	200 )black
	201 )come
	202 )came
	203 )reason
	204 )within
	205 )three
	206 )suck
	207 )valu
	208 )anyth
	209 )convers
	210 )plastic
	211 )import
	212 )line
	213 )alway
	214 )noth
	215 )size
	216 )razr
	217 )tool
	218 )hate
	219 )charm
	220 )kind
	221 )clariti
	222 )sure
	223 )especi
	224 )part
	225 )mani
	226 )drain
	227 )plan
	228 )easili
	229 )rang
	230 )differ
	231 )definit
	232 )free
	233 )anoth
	234 )star
	235 )data
	236 )overal
	237 )player
	238 )coupl
	239 )old
	240 )none
	241 )fall
	242 )wife
	243 )belt
	244 )strong
	245 )job
	246 )awesom
	247 )set
	248 )mobil
	249 )instruct
	250 )provid
	251 )function
	252 )care
	253 )die
	254 )fail
	255 )cingular
	256 )construct
	257 )mic
	258 )effect
	259 )decent
	260 )goe
	261 )face
	262 )must
	263 )scratch
	264 )obvious
	265 )notic
	266 )either
	267 )kept
	268 )flaw
	269 )away
	270 )blue
	271 )rather
	272 )unreli
	273 )nois
	274 )other
	275 )lightweight
	276 )support
	277 )absolut
	278 )actual
	279 )worth
	280 )packag
	281 )fami
	282 )pocket
	283 )access
	284 )seller
	285 )bargain
	286 )plantron
	287 )lock
	288 )weak
	289 )later
	290 )ago
	291 )keyboard
	292 )usb
	293 )beep
	294 )left
	295 )deal
	296 )easier
	297 )store
	298 )palm
	299 )leather
	300 )fast
	301 )satisfi
	302 )ring
	303 )review
	304 )wrong
	305 )extra
	306 )mistak
	307 )choic
	308 )avoid
	309 )treo
	310 )earbud
	311 )said
	312 )lg
	313 )unfortun
	314 )rington
	315 )glad
	316 )abl
	317 )aw
	318 )allow
	319 )serious
	320 )cut
	321 )ll
	322 )front
	323 )cover
	324 )let
	325 )simp
	326 )outlet
	327 )despit
	328 )tone
	329 )thought
	330 )iphon
	331 )complaint
	332 )video
	333 )ipod
	334 )second
	335 )accept
	336 )tinni
	337 )microphon
	338 )download
	339 )match
	340 )form
	341 )push
	342 )incred
	343 )oper
	344 )rate
	345 )dont
	346 )carri
	347 )save
	348 )touch
	349 )refund
	350 )final
	351 )lack
	352 )yet
	353 )unless
	354 )display
	355 )extrem
	356 )wireless
	357 )instead
	358 )rock
	359 )jawbon
	360 )flawless
	361 )mess
	362 )commun
	363 )maintain
	364 )static
	365 )practic
	366 )uncomfort
	367 )soni
	368 )basic
	369 )though
	370 )number
	371 )keypad
	372 )longer
	373 )lost
	374 )may
	375 )everyon
	376 )happen
	377 )websit
	378 )setup
	379 )plus
	380 )market
	381 )direct
	382 )beauti
	383 )cost
	384 )home
	385 )holster
	386 )normal
	387 )speaker
	388 )given
	389 )super
	390 )listen
	391 )bt
	392 )thank
	393 )perhap
	394 )amaz
	395 )oh
	396 )run
	397 )almost
	398 )describ
	399 )pc
	400 )cellphon
	401 )own
	402 )decis
	403 )addit
	404 )handsfre
	405 )stop
	406 )network
	407 )unaccept
	408 )result
	409 )buyer
	410 )coverag
	411 )sharp
	412 )exchang
	413 )happier
	414 )poorli
	415 )friend
	416 )understand
	417 )area
	418 )give
	419 )forev
	420 )pay
	421 )probabl
	422 )instal
	423 )comput
	424 )dead
	425 )experi
	426 )descript
	427 )that
	428 )eas
	429 )secur
	430 )read
	431 )expens
	432 )troubl
	433 )wasn
	434 )appear
	435 )slow
	436 )sprint
	437 )wire
	438 )play
	439 )messag
	440 )current
	441 )flip
	442 )defect
	443 )caus
	444 )includ
	445 )book
	446 )might
	447 )worthless
	448 )fantast
	449 )state
	450 )togeth
	451 )feet
	452 )major
	453 )reach
	454 )dozen
	455 )storag
	456 )buzz
	457 )overrid
	458 )answer
	459 )laptop
	460 )extend
	461 )insid
	462 )mak
	463 )lose
	464 )ok
	465 )check
	466 )loos
	467 )tooth
	468 )advis
	469 )rest
	470 )control
	471 )wonder
	472 )wow
	473 )thin
	474 )tick
	475 )fire
	476 )nearli
	477 )bother
	478 )room
	479 )issu
	480 )felt
	481 )pull
	482 )earphon
	483 )crack
	484 )embarrass
	485 )consum
	486 )background
	487 )unus
	488 )certainli
	489 )usual
	490 )bit
	491 )tell
	492 )excit
	493 )gel
	494 )whatsoev
	495 )least
	496 )purpos
	497 )smell
	498 )flimsi
	499 )revers
	500 )whole
	501 )ador
	502 )wise
	503 )complain
	504 )regard
	505 )compar
	506 )gotten
	507 )pda
	508 )driv
	509 )dial
	510 )cant
	511 )gadget
	512 )larg
	513 )neither
	514 )game
	515 )essenti
	516 )forget
	517 )switch
	518 )tech
	519 )rip
	520 )recharg
	521 )particular
	522 )cradl
	523 )parti
	524 )clearli
	525 )improv
	526 )mp3
	527 )along
	528 )auto
	529 )skype
	530 )activ
	531 )sudden
	532 )glass
	533 )sometim
	534 )mention
	535 )exact
	536 )seri
	537 )wait
	538 )quiet
	539 )person
	540 )dock
	541 )station
	542 )d807
	543 )advertis
	544 )handi
	545 )stupid
	546 )att
	547 )note
	548 )cheaper
	549 )model
	550 )warn
	551 )alon
	552 )music
	553 )moto
	554 )figur
	555 )key
	556 )bewar
	557 )pad
	558 )sunglass
	559 )procedur
	560 )cumbersom
	561 )pros
	562 )worthwhil
	563 )white
	564 )huge
	565 )won
	566 )regret
	567 )although
	568 )user
	569 )abil
	570 )im
	571 )cancel
	572 )resolut
	573 )catch
	574 )ask
	575 )slim
	576 )timefram
	577 )sex
	578 )sourc
	579 )slid
	580 )refus
	581 )sleek
	582 )accident
	583 )window
	584 )full
	585 )took
	586 )unhappi
	587 )eargel
	588 )numer
	589 )total
	590 )done
	591 )continu
	592 )bare
	593 )joke
	594 )forc
	595 )adapt
	596 )re
	597 )logitech
	598 )stuff
	599 )broken
	600 )com
	601 )hous
	602 )pain
	603 )beat
	604 )recognit
	605 )tremend
	606 )experienc
	607 )liter
	608 )apart
	609 )dissapoint
	610 )hop
	611 )link
	612 )blackberri
	613 )brand
	614 )red
	615 )echo
	616 )technolog
	617 )wouldn
	618 )wind
	619 )told
	620 )warranti
	621 )someth
	622 )previous
	623 )utter
	624 )loop
	625 )w810i
	626 )superb
	627 )spring
	628 )period
	629 )third
	630 )flash
	631 )chines
	632 )crisp
	633 )except
	634 )open
	635 )power
	636 )graphic
	637 )wall
	638 )igo
	639 )tip
	640 )etc
	641 )show
	642 )offer
	643 )option
	644 )next
	645 )protector
	646 )date
	647 )bottom
	648 )manag
	649 )convert
	650 )tie
	651 )jiggl
	652 )hundr
	653 )imagin
	654 )fun
	655 )owner
	656 )needless
	657 )seper
	658 )mere
	659 )ft
	660 )excess
	661 )garbl
	662 )odd
	663 )fool
	664 )click
	665 )mechan
	666 )follow
	667 )kindl
	668 )commerci
	669 )mislead
	670 )mother
	671 )combin
	672 )couldnt
	673 )breakag
	674 )ideal
	675 )whose
	676 )sensit
	677 )mov
	678 )freeway
	679 )speed
	680 )contract
	681 )ac
	682 )juic
	683 )highi
	684 )min
	685 )short
	686 )2mp
	687 )pic
	688 )garbag
	689 )mind
	690 )gonna
	691 )argu
	692 )bulki
	693 )usabl
	694 )world
	695 )machin
	696 )neat
	697 )stream
	698 )submerg
	699 )microsoft
	700 )facepl
	701 )eleg
	702 )angl
	703 )drawback
	704 )paus
	705 )skip
	706 )song
	707 )situat
	708 )bmw
	709 )fairli
	710 )everyday
	711 )intend
	712 )boy
	713 )load
	714 )greater
	715 )bud
	716 )waaay
	717 )integr
	718 )seamless
	719 )flush
	720 )toilet
	721 )suppos
	722 )appar
	723 )style
	724 )correct
	725 )jabra350
	726 )megapixel
	727 )render
	728 )imag
	729 )relativeli
	730 )purcash
	731 )geeki
	732 )toast
	733 )ooz
	734 )embed
	735 )stylish
	736 )compromis
	737 )qwerti
	738 )winner
	739 )simpler
	740 )iam
	741 )disapoin
	742 )realiz
	743 )accompani
	744 )brilliant
	745 )damag
	746 )definitli
	747 )peachi
	748 )keen
	749 )upstair
	750 )basem
	751 )reccomend
	752 )relat
	753 )curv
	754 )funni
	755 )sketchi
	756 )web
	757 )brows
	758 )signific
	759 )faster
	760 )build
	761 )unlik
	762 )whine
	763 )monkey
	764 )shouldn
	765 )share
	766 )dna
	767 )copi
	768 )human
	769 )bougth
	770 )l7c
	771 )mode
	772 )file
	773 )browser
	774 )hs850
	775 )whether
	776 )latest
	777 )os
	778 )v1
	779 )15g
	780 )crawl
	781 )recogn
	782 )bluetoooth
	783 )thorn
	784 )abhor
	785 )recent
	786 )disconnect
	787 )buck
	788 )mail
	789 )night
	790 )backlight
	791 )late
	792 )contstruct
	793 )rotat
	794 )forgot
	795 )weird
	796 )iriv
	797 )spinn
	798 )gave
	799 )fond
	800 )magnet
	801 )strap
	802 )psych
	803 )appoint
	804 )giv
	805 )boost
	806 )fri
	807 )bland
	808 )concret
	809 )sanyo
	810 )surviv
	811 )blacktop
	812 )ill
	813 )potenti
	814 )enter
	815 )modest
	816 )cellular
	817 )absolutel
	818 )knock
	819 )wish
	820 )awsom
	821 )earpad
	822 )stereo
	823 )displeas
	824 )scari
	825 )unbear
	826 )risk
	827 )built
	828 )wood
	829 )petroleum
	830 )restor
	831 )jx
	832 )transform
	833 )organiz
	834 )search
	835 )rubber
	836 )capabl
	837 )remov
	838 )lit
	839 )factor
	840 )attract
	841 )portabl
	842 )colleagu
	843 )gosh
	844 )fulli
	845 )bed
	846 )wi
	847 )fi
	848 )morn
	849 )durabl
	850 )memori
	851 )card
	852 )glove
	853 )hat
	854 )sit
	855 )shipment
	856 )solid
	857 )surefir
	858 )gx2
	859 )bt50
	860 )aspect
	861 )remors
	862 )accessoryon
	863 )inexcus
	864 )chang
	865 )carrier
	866 )tmobil
	867 )updat
	868 )vehicl
	869 )ngage
	870 )deliveri
	871 )vx9900
	872 )env
	873 )pleather
	874 )rocket
	875 )destin
	876 )unknown
	877 )condit
	878 )clock
	879 )bill
	880 )excruti
	881 )pric
	882 )overnight
	883 )jerk
	884 )los
	885 )angel
	886 )type
	887 )wallet
	888 )starter
	889 )piti
	890 )respect
	891 )penni
	892 )defeat
	893 )stuck
	894 )max
	895 )mute
	896 )hybrid
	897 )palmtop
	898 )sync
	899 )role
	900 )bt250v
	901 )calendar
	902 )alarm
	903 )bose
	904 )loudspeak
	905 )cute
	906 )nyc
	907 )commut
	908 )shine
	909 )authent
	910 )photo
	911 )ad
	912 )earlier
	913 )favorit
	914 )ericsson
	915 )frog
	916 )eye
	917 )bumper
	918 )aluminum
	919 )hair
	920 )vx
	921 )handheld
	922 )appeal
	923 )immedi
	924 )waterproof
	925 )standard
	926 )leak
	927 )edg
	928 )pant
	929 )headband
	930 )ugli
	931 )shield
	932 )incredi
	933 )hot
	934 )child
	935 )accord
	936 )gentl
	937 )mostli
	938 )threw
	939 )applifi
	940 )manual
	941 )inch
	942 )kitchen
	943 )counter
	944 )laugh
	945 )trunk
	946 )frequently4
	947 )hitch
	948 )ampl
	949 )special
	950 )channel
	951 )increas
	952 )freez
	953 )proper
	954 )miss
	955 )tracfon
	956 )shift
	957 )bubbl
	958 )peel
	959 )droid
	960 )zero
	961 )exercis
	962 )frustrat
	963 )earset
	964 )outgo
	965 )transmiss
	966 )patient
	967 )wiref
	968 )infatu
	969 )inform
	970 )aggrav
	971 )enjoy
	972 )virgin
	973 )muddi
	974 )cas
	975 )insert
	976 )glu
	977 )isn
	978 )plantroninc
	979 )s11
	980 )disapoint
	981 )fourth
	982 )fix
	983 )liv
	984 )roam
	985 )mean
	986 )clever
	987 )constant
	988 )finish
	989 )earbug
	990 )due
	991 )engin
	992 )anyway
	993 )drivng
	994 )walk
	995 )usag
	996 )wip
	997 )strength
	998 )louder
	999 )onlin
	1000 )menus
	1001 )navig
	1002 )recess
	1003 )riington
	1004 )smok
	1005 )lesson
	1006 )effort
	1007 )posses
	1008 )idea
	1009 )trash
	1010 )research
	1011 )develop
	1012 )divis
	1013 )killer
	1014 )cours
	1015 )infuri
	1016 )walkman
	1017 )europ
	1018 )asia
	1019 )deffinit
	1020 )cent
	1021 )tape
	1022 )beh
	1023 )embarass
	1024 )fraction
	1025 )learn
	1026 )crappi
	1027 )e715
	1028 )seeen
	1029 )interfac
	1030 )decad
	1031 )compet
	1032 )dollar
	1033 )700w
	1034 )transmit
	1035 )transceiv
	1036 )steer
	1037 )genuin
	1038 )replacementr
	1039 )pen
	1040 )pack
	1041 )buyit
	1042 )hurt
	1043 )finger
	1044 )good7
	1045 )believ
	1046 )steep
	1047 )point
	1048 )pixel
	1049 )sooner
	1050 )haul
	1051 )averag
	1052 )invent
	1053 )mega
	1054 )discard
	1055 )post
	1056 )detail
	1057 )comment
	1058 )grey
	1059 )frequenty
	1060 )h500
	1061 )guess
	1062 )exist
	1063 )cds
	1064 )surpris
	1065 )fabul
	1066 )transmitt
	1067 )shooter
	1068 )delay
	1069 )bitpim
	1070 )program
	1071 )transfer
	1072 )accessori
	1073 )manufactur
	1074 )fm
	1075 )muffl
	1076 )avail
	1077 )incom
	1078 )soyo
	1079 )self
	1080 )resist
	1081 )over
	1082 )portrait
	1083 )outsid
	1084 )produc
	1085 )receipt
	1086 )luck
	1087 )linksi
	1088 )refurb
	1089 )exterior
	1090 )snug
	1091 )heavi
	1092 )shouldv
	1093 )promis
	1094 )complim
	1095 )tini
	1096 )four
	1097 )someon
	1098 )latch
	1099 )visor
	1100 )address
	1101 )reboot
	1102 )tungsten
	1103 )e2
	1104 )flipphon
	1105 )smooth
	1106 )studi
	1107 )interest
	1108 )sin
	1109 )industri
	1110 )track
	1111 )detach
	1112 )magic
	1113 )somehow
	1114 )upload
	1115 )v3i
	1116 )promptli
	1117 )randomli
	1118 )truli
	1119 )worn
	1120 )ringer
	1121 )electron
	1122 )balanc
	1123 )readi
	1124 )prime
	1125 )upbeat
	1126 )adhes
	1127 )forgeri
	1128 )abound
	1129 )explain
	1130 )jack
	1131 )ca
	1132 )today
	1133 )smallest
	1134 )biggest
	1135 )superfast
	1136 )ergonom
	1137 )theori
	1138 )stand
	1139 )strang
	1140 )occupi
	1141 )distract
	1142 )entir
	1143 )razor
	1144 )cbr
	1145 )mp3s
	1146 )prefer
	1147 )media
	1148 )shot
	1149 )sos
	1150 )mini
	1151 )near
	1152 )destroy
	1153 )startac
	1154 )outperform
	1155 )china
	1156 )v325i
	1157 )sim
	1158 )3o
	1159 )crash
	1160 )invest
	1161 )4s
	1162 )prettier
	1163 )multipl
	1164 )encourag
	1165 )imac
	1166 )extern
	1167 )strip
	1168 )deaf
	1169 )attack
	1170 )elsewher
	1171 )bell
	1172 )whistl
	1173 )mediocr
	1174 )via
	1175 )slide
	1176 )grip
	1177 )prevent
	1178 )slip
	1179 )span
	1180 )exclaim
	1181 )whoa
	1182 )tv
	1183 )cord
	1184 )freedom
	1185 )pass
	1186 )mark
	1187 )inexpens
	1188 )sign
	1189 )soft
	1190 )tight
	1191 )weight
	1192 )shape
	1193 )copier
	1194 )sent
	1195 )anywher
	1196 )sold
	1197 )classi
	1198 )krussel
	1199 )tracfonewebsit
	1200 )toactiv
	1201 )texa
	1202 )dit
	1203 )mainli
	1204 )soon
	1205 )whatev
	1206 )reciev
	1207 )blueant
	1208 )supertooth
	1209 )metro
	1210 )pcs
	1211 )sch
	1212 )r450
	1213 )slider
	1214 )premium
	1215 )plenti
	1216 )capac
	1217 )confort
	1218 )somewhat
	1219 )ant
	1220 )hey
	1221 )pleasant
	1222 )supris
	1223 )dustpan
	1224 )indoor
	1225 )dispos
	1226 )puff
	1227 )smoke
	1228 )conveni
	1229 )ride
	1230 )smoother
	1231 )nano
	1232 )son
	1233 )reccommend
	1234 )highest
	1235 )anti
	1236 )glare
	1237 )prompt
	1238 )cat
	1239 )smartphon
	1240 )wont
	1241 )atleast
	1242 )amp
	1243 )reoccur
	1244 )antena
	1245 )somewher
	1246 )els
	1247 )creak
	1248 )wooden
	1249 )floor
	1250 )gener
	1251 )inconspicu
	1252 )boot
	1253 )slowli
	1254 )sorri
	1255 )imposs
	1256 )upgrad
	1257 )discount
	1258 )securli
	1259 )possibl
	1260 )doubl
	1261 )entertain
	1262 )handset
	1263 )activesync
	1264 )optim
	1265 )synchron
	1266 )disgust
	1267 )coupon
	1268 )rare
	1269 )instanc
	1270 )ps3
	1271 )five
	1272 )cheapi
	1273 )shout
	1274 )telephon
	1275 )yes
	1276 )shini
	1277 )grtting
	1278 )v3c
	1279 )thumb
	1280 )kit
	1281 )exceed
	1282 )thru
	1283 )sight
	1284 )improp
	1285 )everywher
	1286 )awkward
	1287 )hope
	1288 )father
	1289 )v265
	1290 )hit
	1291 )intermitt
	1292 )cingulair
	1293 )row
	1294 )nightmar
	1295 )speakerphon
	1296 )cassett
	1297 )dirti
	1298 )nicer
	1299 )haven
	1300 )sensor
	1301 )reliabl
	1302 )wit
	1303 )era
	1304 )ir
	1305 )counterfeit
	1306 )see
	1307 )travl
	1308 )swivel
	1309 )sister
	1310 )dual
	1311 )overnit
	1312 )bottowm
	1313 )gimmick
	1314 )top
	1315 )discomfort
	1316 )trust
	1317 )hoursth
	1318 )confus
	1319 )thereplac
	1320 )holder
	1321 )cutout
	1322 )land
	1323 )materi
	1324 )offici
	1325 )oem
	1326 )loudest
	1327 )competitor
	1328 )alot
	1329 )cheapli
	1330 )unintellig
	1331 )word
	1332 )restart
	1333 )bend
	1334 )leaf
	1335 )metal
	1336 )stress
	1337 )leopard
	1338 )print
	1339 )wild
	1340 )saggi
	1341 )floppi
	1342 )add
	1343 )snap
	1344 )fliptop
	1345 )wobbl
	1346 )eventu
	1347 )seat
	1348 )fulfil
	1349 )requir
	1350 )fact
	1351 )distort
	1352 )lap
	1353 )yell
	1354 )mine
	1355 )christma
	1356 )otherwis
	1357 )joy
	1358 )satisif
	1359 )s710a
	1360 )hing
	1361 )spec
	1362 )armband
	1363 )allot
	1364 )clearer
	1365 )ericson
	1366 )z500a
	1367 )motor
	1368 )center
	1369 )voltag
	1370 )hum
	1371 )equip
	1372 )certain
	1373 )girl
	1374 )wake
	1375 )styl
	1376 )restock
	1377 )fee
	1378 )darn
	1379 )lousi
	1380 )seen
	1381 )sweetest
	1382 )hook
	1383 )canal
	1384 )unsatisfactori
	1385 )negativeli
	1386 )hype
	1387 )assum
	1388 )lens
	1389 )text
	1390 )tricki
	1391 )blew
	1392 )flop
	1393 )smudg
	1394 )infra
	1395 )port
	1396 )irda

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
		outputs[14] = (inputs[14]-0.03299999982)/0.1787258834
		outputs[15] = (inputs[15]-0.03299999982)/0.1842415333
		outputs[16] = (inputs[16]-0.03200000152)/0.1871123016
		outputs[17] = (inputs[17]-0.03200000152)/0.1760880649
		outputs[18] = (inputs[18]-0.02899999917)/0.1678903997
		outputs[19] = (inputs[19]-0.02800000086)/0.1767689139
		outputs[20] = (inputs[20]-0.02800000086)/0.1710124165
		outputs[21] = (inputs[21]-0.02800000086)/0.1823437661
		outputs[22] = (inputs[22]-0.0270000007)/0.1621644199
		outputs[23] = (inputs[23]-0.02600000054)/0.1592147946
		outputs[24] = (inputs[24]-0.02500000037)/0.1685330868
		outputs[25] = (inputs[25]-0.02500000037)/0.1562030762
		outputs[26] = (inputs[26]-0.02500000037)/0.162485078
		outputs[27] = (inputs[27]-0.02400000021)/0.1595288366
		outputs[28] = (inputs[28]-0.02400000021)/0.153125599
		outputs[29] = (inputs[29]-0.02300000004)/0.156510368
		outputs[30] = (inputs[30]-0.02300000004)/0.1499783099
		outputs[31] = (inputs[31]-0.02300000004)/0.1499783099
		outputs[32] = (inputs[32]-0.02300000004)/0.1499783099
		outputs[33] = (inputs[33]-0.02300000004)/0.1499783099
		outputs[34] = (inputs[34]-0.02300000004)/0.1499783099
		outputs[35] = (inputs[35]-0.02300000004)/0.1499783099
		outputs[36] = (inputs[36]-0.02300000004)/0.156510368
		outputs[37] = (inputs[37]-0.02199999988)/0.1659624726
		outputs[38] = (inputs[38]-0.02199999988)/0.1467567235
		outputs[39] = (inputs[39]-0.02099999972)/0.1434558481
		outputs[40] = (inputs[40]-0.02099999972)/0.1434558481
		outputs[41] = (inputs[41]-0.02099999972)/0.1502716988
		outputs[42] = (inputs[42]-0.01999999955)/0.140070051
		outputs[43] = (inputs[43]-0.01999999955)/0.1470429301
		outputs[44] = (inputs[44]-0.01899999939)/0.1437346786
		outputs[45] = (inputs[45]-0.01899999939)/0.1365930438
		outputs[46] = (inputs[46]-0.01899999939)/0.1365930438
		outputs[47] = (inputs[47]-0.01899999939)/0.1365930438
		outputs[48] = (inputs[48]-0.01899999939)/0.1365930438
		outputs[49] = (inputs[49]-0.01799999923)/0.1330176443
		outputs[50] = (inputs[50]-0.01799999923)/0.1403413564
		outputs[51] = (inputs[51]-0.01799999923)/0.1330176443
		outputs[52] = (inputs[52]-0.01700000092)/0.1293357164
		outputs[53] = (inputs[53]-0.01700000092)/0.1293357164
		outputs[54] = (inputs[54]-0.01700000092)/0.1368566006
		outputs[55] = (inputs[55]-0.01700000092)/0.1293357164
		outputs[56] = (inputs[56]-0.01700000092)/0.1293357164
		outputs[57] = (inputs[57]-0.01700000092)/0.1293357164
		outputs[58] = (inputs[58]-0.01600000076)/0.1255378872
		outputs[59] = (inputs[59]-0.01499999966)/0.121613279
		outputs[60] = (inputs[60]-0.01499999966)/0.121613279
		outputs[61] = (inputs[61]-0.01499999966)/0.1295831501
		outputs[62] = (inputs[62]-0.01499999966)/0.1295831501
		outputs[63] = (inputs[63]-0.01499999966)/0.121613279
		outputs[64] = (inputs[64]-0.01499999966)/0.121613279
		outputs[65] = (inputs[65]-0.01499999966)/0.1295831501
		outputs[66] = (inputs[66]-0.01400000043)/0.1175492108
		outputs[67] = (inputs[67]-0.01400000043)/0.1175492108
		outputs[68] = (inputs[68]-0.01400000043)/0.1175492108
		outputs[69] = (inputs[69]-0.01400000043)/0.1175492108
		outputs[70] = (inputs[70]-0.01400000043)/0.1175492108
		outputs[71] = (inputs[71]-0.01400000043)/0.1175492108
		outputs[72] = (inputs[72]-0.01400000043)/0.1175492108
		outputs[73] = (inputs[73]-0.01400000043)/0.1175492108
		outputs[74] = (inputs[74]-0.01400000043)/0.125776872
		outputs[75] = (inputs[75]-0.01300000027)/0.1218435317
		outputs[76] = (inputs[76]-0.01300000027)/0.1133306846
		outputs[77] = (inputs[77]-0.01300000027)/0.1133306846
		outputs[78] = (inputs[78]-0.01300000027)/0.1133306846
		outputs[79] = (inputs[79]-0.01300000027)/0.1133306846
		outputs[80] = (inputs[80]-0.01300000027)/0.1133306846
		outputs[81] = (inputs[81]-0.01300000027)/0.1133306846
		outputs[82] = (inputs[82]-0.01300000027)/0.1297992617
		outputs[83] = (inputs[83]-0.01300000027)/0.1133306846
		outputs[84] = (inputs[84]-0.01300000027)/0.1133306846
		outputs[85] = (inputs[85]-0.01300000027)/0.1133306846
		outputs[86] = (inputs[86]-0.0120000001)/0.1089397445
		outputs[87] = (inputs[87]-0.0120000001)/0.1089397445
		outputs[88] = (inputs[88]-0.0120000001)/0.1089397445
		outputs[89] = (inputs[89]-0.0120000001)/0.1089397445
		outputs[90] = (inputs[90]-0.0120000001)/0.1089397445
		outputs[91] = (inputs[91]-0.0120000001)/0.1089397445
		outputs[92] = (inputs[92]-0.01099999994)/0.1043546349
		outputs[93] = (inputs[93]-0.01099999994)/0.1135424674
		outputs[94] = (inputs[94]-0.01099999994)/0.1043546349
		outputs[95] = (inputs[95]-0.01099999994)/0.1043546349
		outputs[96] = (inputs[96]-0.01099999994)/0.1043546349
		outputs[97] = (inputs[97]-0.01099999994)/0.1043546349
		outputs[98] = (inputs[98]-0.009999999776)/0.09954853356
		outputs[99] = (inputs[99]-0.009999999776)/0.09954853356
		outputs[100] = (inputs[100]-0.009999999776)/0.09954853356
		outputs[101] = (inputs[101]-0.009999999776)/0.09954853356
		outputs[102] = (inputs[102]-0.009999999776)/0.09954853356
		outputs[103] = (inputs[103]-0.009999999776)/0.09954853356
		outputs[104] = (inputs[104]-0.009999999776)/0.09954853356
		outputs[105] = (inputs[105]-0.009999999776)/0.09954853356
		outputs[106] = (inputs[106]-0.009999999776)/0.09954853356
		outputs[107] = (inputs[107]-0.009999999776)/0.09954853356
		outputs[108] = (inputs[108]-0.009999999776)/0.09954853356
		outputs[109] = (inputs[109]-0.009999999776)/0.1091417074
		outputs[110] = (inputs[110]-0.009999999776)/0.09954853356
		outputs[111] = (inputs[111]-0.008999999613)/0.09448771179
		outputs[112] = (inputs[112]-0.008999999613)/0.09448771179
		outputs[113] = (inputs[113]-0.008999999613)/0.09448771179
		outputs[114] = (inputs[114]-0.008999999613)/0.09448771179
		outputs[115] = (inputs[115]-0.008999999613)/0.09448771179
		outputs[116] = (inputs[116]-0.008999999613)/0.1045463085
		outputs[117] = (inputs[117]-0.008999999613)/0.09448771179
		outputs[118] = (inputs[118]-0.008999999613)/0.09448771179
		outputs[119] = (inputs[119]-0.008999999613)/0.09448771179
		outputs[120] = (inputs[120]-0.008999999613)/0.1045463085
		outputs[121] = (inputs[121]-0.008999999613)/0.1045463085
		outputs[122] = (inputs[122]-0.008999999613)/0.09448771179
		outputs[123] = (inputs[123]-0.008999999613)/0.09448771179
		outputs[124] = (inputs[124]-0.008999999613)/0.09448771179
		outputs[125] = (inputs[125]-0.008999999613)/0.09448771179
		outputs[126] = (inputs[126]-0.008999999613)/0.09448771179
		outputs[127] = (inputs[127]-0.008999999613)/0.09448771179
		outputs[128] = (inputs[128]-0.00800000038)/0.08912880719
		outputs[129] = (inputs[129]-0.00800000038)/0.08912880719
		outputs[130] = (inputs[130]-0.00800000038)/0.08912880719
		outputs[131] = (inputs[131]-0.00800000038)/0.08912880719
		outputs[132] = (inputs[132]-0.00800000038)/0.08912880719
		outputs[133] = (inputs[133]-0.00800000038)/0.08912880719
		outputs[134] = (inputs[134]-0.00800000038)/0.08912880719
		outputs[135] = (inputs[135]-0.00800000038)/0.08912880719
		outputs[136] = (inputs[136]-0.00800000038)/0.08912880719
		outputs[137] = (inputs[137]-0.00800000038)/0.08912880719
		outputs[138] = (inputs[138]-0.00800000038)/0.08912880719
		outputs[139] = (inputs[139]-0.00800000038)/0.0997293666
		outputs[140] = (inputs[140]-0.00800000038)/0.0997293666
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
		outputs[165] = (inputs[165]-0.007000000216)/0.08341437578
		outputs[166] = (inputs[166]-0.007000000216)/0.09465706348
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
		outputs[180] = (inputs[180]-0.006000000052)/0.08928590268
		outputs[181] = (inputs[181]-0.006000000052)/0.07726558298
		outputs[182] = (inputs[182]-0.006000000052)/0.07726558298
		outputs[183] = (inputs[183]-0.006000000052)/0.07726558298
		outputs[184] = (inputs[184]-0.006000000052)/0.07726558298
		outputs[185] = (inputs[185]-0.006000000052)/0.07726558298
		outputs[186] = (inputs[186]-0.006000000052)/0.07726558298
		outputs[187] = (inputs[187]-0.006000000052)/0.07726558298
		outputs[188] = (inputs[188]-0.006000000052)/0.07726558298
		outputs[189] = (inputs[189]-0.006000000052)/0.07726558298
		outputs[190] = (inputs[190]-0.006000000052)/0.07726558298
		outputs[191] = (inputs[191]-0.006000000052)/0.07726558298
		outputs[192] = (inputs[192]-0.006000000052)/0.07726558298
		outputs[193] = (inputs[193]-0.006000000052)/0.07726558298
		outputs[194] = (inputs[194]-0.006000000052)/0.07726558298
		outputs[195] = (inputs[195]-0.006000000052)/0.08928590268
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
		outputs[215] = (inputs[215]-0.004999999888)/0.07056897134
		outputs[216] = (inputs[216]-0.004999999888)/0.07056897134
		outputs[217] = (inputs[217]-0.004999999888)/0.07056897134
		outputs[218] = (inputs[218]-0.004999999888)/0.07056897134
		outputs[219] = (inputs[219]-0.004999999888)/0.07056897134
		outputs[220] = (inputs[220]-0.004999999888)/0.07056897134
		outputs[221] = (inputs[221]-0.004999999888)/0.07056897134
		outputs[222] = (inputs[222]-0.004999999888)/0.07056897134
		outputs[223] = (inputs[223]-0.004999999888)/0.07056897134
		outputs[224] = (inputs[224]-0.004999999888)/0.08355825394
		outputs[225] = (inputs[225]-0.004999999888)/0.07056897134
		outputs[226] = (inputs[226]-0.004999999888)/0.07056897134
		outputs[227] = (inputs[227]-0.004999999888)/0.07056897134
		outputs[228] = (inputs[228]-0.004999999888)/0.07056897134
		outputs[229] = (inputs[229]-0.004999999888)/0.07056897134
		outputs[230] = (inputs[230]-0.004999999888)/0.07056897134
		outputs[231] = (inputs[231]-0.004999999888)/0.07056897134
		outputs[232] = (inputs[232]-0.004999999888)/0.07056897134
		outputs[233] = (inputs[233]-0.004999999888)/0.07056897134
		outputs[234] = (inputs[234]-0.004999999888)/0.08355825394
		outputs[235] = (inputs[235]-0.004999999888)/0.07056897134
		outputs[236] = (inputs[236]-0.004999999888)/0.07056897134
		outputs[237] = (inputs[237]-0.004999999888)/0.07056897134
		outputs[238] = (inputs[238]-0.004999999888)/0.07056897134
		outputs[239] = (inputs[239]-0.004999999888)/0.07056897134
		outputs[240] = (inputs[240]-0.004999999888)/0.07056897134
		outputs[241] = (inputs[241]-0.004999999888)/0.08355825394
		outputs[242] = (inputs[242]-0.004999999888)/0.07056897134
		outputs[243] = (inputs[243]-0.004999999888)/0.07056897134
		outputs[244] = (inputs[244]-0.004999999888)/0.07056897134
		outputs[245] = (inputs[245]-0.004999999888)/0.07056897134
		outputs[246] = (inputs[246]-0.004999999888)/0.07056897134
		outputs[247] = (inputs[247]-0.004999999888)/0.07056897134
		outputs[248] = (inputs[248]-0.004999999888)/0.07056897134
		outputs[249] = (inputs[249]-0.004999999888)/0.07056897134
		outputs[250] = (inputs[250]-0.004999999888)/0.07056897134
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
		outputs[286] = (inputs[286]-0.00400000019)/0.07739502192
		outputs[287] = (inputs[287]-0.00400000019)/0.06315051764
		outputs[288] = (inputs[288]-0.00400000019)/0.06315051764
		outputs[289] = (inputs[289]-0.00400000019)/0.06315051764
		outputs[290] = (inputs[290]-0.00400000019)/0.06315051764
		outputs[291] = (inputs[291]-0.00400000019)/0.06315051764
		outputs[292] = (inputs[292]-0.00400000019)/0.0999699682
		outputs[293] = (inputs[293]-0.00400000019)/0.06315051764
		outputs[294] = (inputs[294]-0.00400000019)/0.06315051764
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
		outputs[314] = (inputs[314]-0.00400000019)/0.06315051764
		outputs[315] = (inputs[315]-0.00400000019)/0.06315051764
		outputs[316] = (inputs[316]-0.00400000019)/0.06315051764
		outputs[317] = (inputs[317]-0.00400000019)/0.06315051764
		outputs[318] = (inputs[318]-0.003000000026)/0.05471740291
		outputs[319] = (inputs[319]-0.003000000026)/0.05471740291
		outputs[320] = (inputs[320]-0.003000000026)/0.05471740291
		outputs[321] = (inputs[321]-0.003000000026)/0.05471740291
		outputs[322] = (inputs[322]-0.003000000026)/0.05471740291
		outputs[323] = (inputs[323]-0.003000000026)/0.05471740291
		outputs[324] = (inputs[324]-0.003000000026)/0.05471740291
		outputs[325] = (inputs[325]-0.003000000026)/0.05471740291
		outputs[326] = (inputs[326]-0.003000000026)/0.05471740291
		outputs[327] = (inputs[327]-0.003000000026)/0.05471740291
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
		outputs[340] = (inputs[340]-0.003000000026)/0.05471740291
		outputs[341] = (inputs[341]-0.003000000026)/0.05471740291
		outputs[342] = (inputs[342]-0.003000000026)/0.05471740291
		outputs[343] = (inputs[343]-0.003000000026)/0.05471740291
		outputs[344] = (inputs[344]-0.003000000026)/0.05471740291
		outputs[345] = (inputs[345]-0.003000000026)/0.05471740291
		outputs[346] = (inputs[346]-0.003000000026)/0.05471740291
		outputs[347] = (inputs[347]-0.003000000026)/0.07068236172
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
		outputs[361] = (inputs[361]-0.003000000026)/0.07068236172
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
		outputs[378] = (inputs[378]-0.003000000026)/0.07068236172
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
		outputs[409] = (inputs[409]-0.003000000026)/0.07068236172
		outputs[410] = (inputs[410]-0.003000000026)/0.05471740291
		outputs[411] = (inputs[411]-0.003000000026)/0.05471740291
		outputs[412] = (inputs[412]-0.003000000026)/0.05471740291
		outputs[413] = (inputs[413]-0.003000000026)/0.05471740291
		outputs[414] = (inputs[414]-0.003000000026)/0.05471740291
		outputs[415] = (inputs[415]-0.003000000026)/0.05471740291
		outputs[416] = (inputs[416]-0.003000000026)/0.05471740291
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
		outputs[443] = (inputs[443]-0.003000000026)/0.07068236172
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
		outputs[458] = (inputs[458]-0.001000000047)/0.0316227749
		outputs[459] = (inputs[459]-0.002000000095)/0.04469897225
		outputs[460] = (inputs[460]-0.002000000095)/0.04469897225
		outputs[461] = (inputs[461]-0.002000000095)/0.04469897225
		outputs[462] = (inputs[462]-0.002000000095)/0.04469897225
		outputs[463] = (inputs[463]-0.001000000047)/0.0316227749
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
		outputs[476] = (inputs[476]-0.002000000095)/0.04469897225
		outputs[477] = (inputs[477]-0.002000000095)/0.04469897225
		outputs[478] = (inputs[478]-0.002000000095)/0.04469897225
		outputs[479] = (inputs[479]-0.002000000095)/0.04469897225
		outputs[480] = (inputs[480]-0.002000000095)/0.04469897225
		outputs[481] = (inputs[481]-0.002000000095)/0.04469897225
		outputs[482] = (inputs[482]-0.002000000095)/0.04469897225
		outputs[483] = (inputs[483]-0.001000000047)/0.0316227749
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
		outputs[535] = (inputs[535]-0.002000000095)/0.04469897225
		outputs[536] = (inputs[536]-0.002000000095)/0.04469897225
		outputs[537] = (inputs[537]-0.002000000095)/0.04469897225
		outputs[538] = (inputs[538]-0.002000000095)/0.04469897225
		outputs[539] = (inputs[539]-0.002000000095)/0.04469897225
		outputs[540] = (inputs[540]-0.002000000095)/0.04469897225
		outputs[541] = (inputs[541]-0.001000000047)/0.0316227749
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
		outputs[576] = (inputs[576]-0.001000000047)/0.0316227749
		outputs[577] = (inputs[577]-0.002000000095)/0.04469897225
		outputs[578] = (inputs[578]-0.002000000095)/0.04469897225
		outputs[579] = (inputs[579]-0.002000000095)/0.04469897225
		outputs[580] = (inputs[580]-0.002000000095)/0.04469897225
		outputs[581] = (inputs[581]-0.001000000047)/0.0316227749
		outputs[582] = (inputs[582]-0.002000000095)/0.04469897225
		outputs[583] = (inputs[583]-0.002000000095)/0.04469897225
		outputs[584] = (inputs[584]-0.002000000095)/0.04469897225
		outputs[585] = (inputs[585]-0.002000000095)/0.04469897225
		outputs[586] = (inputs[586]-0.002000000095)/0.04469897225
		outputs[587] = (inputs[587]-0.002000000095)/0.04469897225
		outputs[588] = (inputs[588]-0.002000000095)/0.04469897225
		outputs[589] = (inputs[589]-0.002000000095)/0.04469897225
		outputs[590] = (inputs[590]-0.002000000095)/0.04469897225
		outputs[591] = (inputs[591]-0.002000000095)/0.04469897225
		outputs[592] = (inputs[592]-0.002000000095)/0.04469897225
		outputs[593] = (inputs[593]-0.002000000095)/0.04469897225
		outputs[594] = (inputs[594]-0.002000000095)/0.04469897225
		outputs[595] = (inputs[595]-0.002000000095)/0.04469897225
		outputs[596] = (inputs[596]-0.002000000095)/0.04469897225
		outputs[597] = (inputs[597]-0.002000000095)/0.04469897225
		outputs[598] = (inputs[598]-0.002000000095)/0.04469897225
		outputs[599] = (inputs[599]-0.002000000095)/0.04469897225
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
		outputs[614] = (inputs[614]-0.002000000095)/0.04469897225
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

		combinations[0] = -0.0602796 -0.065373*inputs[0] -0.0932928*inputs[1] -0.213813*inputs[2] -0.151831*inputs[3] -0.151025*inputs[4] -0.0765797*inputs[5] -0.0125552*inputs[6] -0.0281329*inputs[7] -0.0496826*inputs[8] -0.00184195*inputs[9] -0.00150644*inputs[10] -0.0736489*inputs[11] -0.0355652*inputs[12] -0.0973336*inputs[13] -0.181973*inputs[14] -0.0384024*inputs[15] -0.00129846*inputs[16] -0.0189505*inputs[17] -0.108198*inputs[18] +0.109633*inputs[19] +0.00103601*inputs[20] -0.0145884*inputs[21] -0.0282878*inputs[22] -0.000683286*inputs[23] +0.0047133*inputs[24] +0.024515*inputs[25] -0.125869*inputs[26] +0.0234616*inputs[27] -0.00402065*inputs[28] -0.00640616*inputs[29] -0.0157121*inputs[30] +0.0336065*inputs[31] +0.0982767*inputs[32] -0.0743425*inputs[33] +0.0831569*inputs[34] -0.103392*inputs[35] -0.0868831*inputs[36] -0.00836014*inputs[37] +0.0108126*inputs[38] +0.131595*inputs[39] -0.0316417*inputs[40] -0.287095*inputs[41] -0.0111365*inputs[42] +0.0416874*inputs[43] -0.0353546*inputs[44] -0.0018677*inputs[45] +0.0260705*inputs[46] +0.0685641*inputs[47] +0.0809638*inputs[48] -0.0912773*inputs[49] -0.003245*inputs[50] -0.00667989*inputs[51] -0.00598694*inputs[52] -0.0157339*inputs[53] +0.00849036*inputs[54] +0.0607745*inputs[55] -0.0126274*inputs[56] -0.0351472*inputs[57] +0.120569*inputs[58] +0.00155285*inputs[59] +0.140608*inputs[60] +0.0293267*inputs[61] +0.0301908*inputs[62] +0.0387983*inputs[63] -0.000766496*inputs[64] +0.157862*inputs[65] +0.0853792*inputs[66] -0.0482011*inputs[67] +0.0425566*inputs[68] -0.00018297*inputs[69] -0.0102574*inputs[70] +0.00327068*inputs[71] -0.0118094*inputs[72] +0.011017*inputs[73] +0.0903944*inputs[74] -0.0117184*inputs[75] -0.101008*inputs[76] -0.0171878*inputs[77] -0.0206052*inputs[78] -0.0121986*inputs[79] +0.0334327*inputs[80] +0.0407921*inputs[81] -0.0129555*inputs[82] -0.0223643*inputs[83] -0.0197727*inputs[84] +0.0449212*inputs[85] -0.0438211*inputs[86] +0.0168045*inputs[87] +0.0339883*inputs[88] +0.0147711*inputs[89] +0.00752763*inputs[90] +0.0247567*inputs[91] +0.0342628*inputs[92] +0.0225498*inputs[93] +0.106308*inputs[94] +0.0892108*inputs[95] +0.00226799*inputs[96] -0.032883*inputs[97] +0.0156068*inputs[98] -0.0380844*inputs[99] -0.00634938*inputs[100] +0.0331771*inputs[101] +0.0221179*inputs[102] +0.01441*inputs[103] +0.0490593*inputs[104] -0.00306321*inputs[105] +0.0816184*inputs[106] -0.00259229*inputs[107] +0.0855128*inputs[108] -0.00839884*inputs[109] -0.0448492*inputs[110] -0.0590383*inputs[111] +0.0207497*inputs[112] -0.0201925*inputs[113] +0.0641337*inputs[114] -0.0192021*inputs[115] +0.00424027*inputs[116] +0.0172612*inputs[117] +0.0339453*inputs[118] +0.0179441*inputs[119] +0.0324368*inputs[120] -0.0792016*inputs[121] -0.0707673*inputs[122] +0.0854895*inputs[123] +0.0293114*inputs[124] +0.013601*inputs[125] +0.00377777*inputs[126] +0.0035773*inputs[127] -0.00677585*inputs[128] -0.00418431*inputs[129] -0.00831501*inputs[130] -0.119452*inputs[131] -0.0209093*inputs[132] -0.0113858*inputs[133] +0.0219461*inputs[134] +0.0257582*inputs[135] +0.0804436*inputs[136] +0.00576605*inputs[137] +0.0225512*inputs[138] +0.0869383*inputs[139] -0.0130172*inputs[140] -0.022463*inputs[141] -0.0287541*inputs[142] -0.0286632*inputs[143] +0.0153425*inputs[144] -0.0102547*inputs[145] -0.00290921*inputs[146] +0.0189877*inputs[147] +0.02621*inputs[148] +0.0394235*inputs[149] +0.0195128*inputs[150] +0.0152777*inputs[151] +0.0369445*inputs[152] +0.00144461*inputs[153] +0.0659262*inputs[154] +0.0177104*inputs[155] -0.00160626*inputs[156] +0.00982063*inputs[157] +0.0259958*inputs[158] -0.0635281*inputs[159] +0.0046985*inputs[160] +0.0143443*inputs[161] +1.25448e-05*inputs[162] +0.0245122*inputs[163] +0.00804437*inputs[164] +0.0177671*inputs[165] -0.0201066*inputs[166] -0.0064922*inputs[167] +0.0010468*inputs[168] +0.0222948*inputs[169] +0.0113266*inputs[170] +0.0388962*inputs[171] -0.00514445*inputs[172] -0.0215674*inputs[173] +0.0166909*inputs[174] -0.0520281*inputs[175] -0.0113756*inputs[176] -0.00245886*inputs[177] +0.0722323*inputs[178] -0.0440961*inputs[179] +0.0574624*inputs[180] -0.00245957*inputs[181] -0.00122885*inputs[182] -0.012256*inputs[183] +0.0130259*inputs[184] +0.00270405*inputs[185] +0.012356*inputs[186] +0.0123256*inputs[187] -0.0154114*inputs[188] +0.00917095*inputs[189] +0.0255293*inputs[190] -0.0047931*inputs[191] -0.0348829*inputs[192] +0.00706691*inputs[193] -0.0508894*inputs[194] -0.0191905*inputs[195] -0.0199905*inputs[196] +0.0375355*inputs[197] -0.0345637*inputs[198] +0.021623*inputs[199] +0.000379694*inputs[200] -0.00408678*inputs[201] +0.00443267*inputs[202] +0.0144702*inputs[203] -0.0264757*inputs[204] +0.0427758*inputs[205] -0.0226832*inputs[206] +0.0268962*inputs[207] +0.0182733*inputs[208] +0.0539364*inputs[209] -0.000400361*inputs[210] +0.00583895*inputs[211] +0.0279689*inputs[212] +0.00351408*inputs[213] -0.0174092*inputs[214] -0.0553191*inputs[215] -0.0268392*inputs[216] +0.0223778*inputs[217] -0.00960851*inputs[218] +0.00725477*inputs[219] -0.0170836*inputs[220] +0.0183169*inputs[221] -0.0151293*inputs[222] -0.00293847*inputs[223] +0.0438945*inputs[224] +0.0562697*inputs[225] +0.000553777*inputs[226] +0.00826664*inputs[227] -0.00981101*inputs[228] -0.0170958*inputs[229] -0.0192778*inputs[230] -0.0040105*inputs[231] +0.00452629*inputs[232] -0.00929981*inputs[233] -0.000896465*inputs[234] -0.0395522*inputs[235] +0.00158648*inputs[236] -0.0179254*inputs[237] +0.0116868*inputs[238] +0.0424735*inputs[239] +0.00426974*inputs[240] +0.00246267*inputs[241] +0.00631092*inputs[242] +0.0186236*inputs[243] +0.0136617*inputs[244] -0.0352405*inputs[245] +0.018558*inputs[246] +0.00459332*inputs[247] +0.0487509*inputs[248] +0.00809451*inputs[249] -0.00775485*inputs[250] +0.0292876*inputs[251] +0.0137984*inputs[252] +0.0373503*inputs[253] +0.0160851*inputs[254] +0.00840821*inputs[255] -0.0124379*inputs[256] +0.00307463*inputs[257] -0.00951395*inputs[258] +0.00559643*inputs[259] +0.003817*inputs[260] -0.0299145*inputs[261] -0.0029657*inputs[262] +0.00449968*inputs[263] -0.00289178*inputs[264] +0.0108787*inputs[265] -0.0256703*inputs[266] +0.0606573*inputs[267] +0.0233117*inputs[268] -0.0110421*inputs[269] +0.00917564*inputs[270] +0.0554935*inputs[271] +0.0048237*inputs[272] -0.00624586*inputs[273] +0.00381699*inputs[274] +0.003817*inputs[275] -0.0103614*inputs[276] +0.00838517*inputs[277] +0.0525989*inputs[278] -4.15413e-05*inputs[279] -0.022171*inputs[280] +0.0255815*inputs[281] -0.0191601*inputs[282] -0.0294356*inputs[283] -0.0209913*inputs[284] -0.0783741*inputs[285] +0.0518223*inputs[286] +0.025995*inputs[287] +0.017206*inputs[288] -0.0205043*inputs[289] -0.0225277*inputs[290] +0.00307132*inputs[291] +0.00913733*inputs[292] +0.0170725*inputs[293] -0.0183712*inputs[294] -0.00296847*inputs[295] +0.0182837*inputs[296] -0.00149901*inputs[297] -0.0201937*inputs[298] -0.0475722*inputs[299] -0.0352318*inputs[300] -0.0105955*inputs[301] -0.00474124*inputs[302] +0.0885654*inputs[303] -0.00643813*inputs[304] +0.042926*inputs[305] +0.0191686*inputs[306] +0.0403155*inputs[307] +0.0195492*inputs[308] +0.0213449*inputs[309] +0.00307463*inputs[310] +0.00875593*inputs[311] +0.0628868*inputs[312] -0.00899179*inputs[313] -0.0406692*inputs[314] -0.0105251*inputs[315] +0.00668374*inputs[316] -0.00190923*inputs[317] -0.00428243*inputs[318] -0.0305913*inputs[319] +0.0282108*inputs[320] +0.00551763*inputs[321] +0.0048686*inputs[322] +0.0171578*inputs[323] +0.0027927*inputs[324] +0.000148813*inputs[325] +0.00648586*inputs[326] -0.00384133*inputs[327] -0.00091479*inputs[328] +0.00330392*inputs[329] +0.0135122*inputs[330] -0.0182085*inputs[331] -0.0292359*inputs[332] +0.00486862*inputs[333] +0.0265181*inputs[334] +0.0242871*inputs[335] +0.00836266*inputs[336] -0.0164817*inputs[337] +0.0428083*inputs[338] +0.00680438*inputs[339] +0.00765133*inputs[340] -0.0485242*inputs[341] +0.00055*inputs[342] -0.010678*inputs[343] +0.0243338*inputs[344] -0.0321987*inputs[345] +0.012979*inputs[346] +0.00255753*inputs[347] +0.0215892*inputs[348] -0.015968*inputs[349] +0.000966431*inputs[350] -0.0391772*inputs[351] +0.0065076*inputs[352] -0.00951952*inputs[353] +0.0137031*inputs[354] -0.00866732*inputs[355] +0.00246078*inputs[356] -0.0115739*inputs[357] +0.00330388*inputs[358] -0.00540059*inputs[359] -0.0146447*inputs[360] +0.000333396*inputs[361] +0.00691175*inputs[362] +0.0135393*inputs[363] -0.0118162*inputs[364] +0.0173868*inputs[365] +0.00119853*inputs[366] +0.0238823*inputs[367] -0.0196987*inputs[368] +0.0151769*inputs[369] +0.00330388*inputs[370] +0.00330387*inputs[371] +0.00988148*inputs[372] +0.0122792*inputs[373] -0.0181425*inputs[374] +0.0108079*inputs[375] -0.00955652*inputs[376] -0.0357926*inputs[377] -0.01152*inputs[378] -0.0179631*inputs[379] -0.00203608*inputs[380] -0.042052*inputs[381] +0.0266859*inputs[382] +0.0133544*inputs[383] +0.00875496*inputs[384] +0.0059804*inputs[385] +0.00479201*inputs[386] -0.00655555*inputs[387] +0.00330388*inputs[388] +0.0156734*inputs[389] +0.00864358*inputs[390] -0.00459311*inputs[391] +0.00330387*inputs[392] -0.0310902*inputs[393] -0.0180032*inputs[394] +0.0423664*inputs[395] -0.00370267*inputs[396] -0.00996978*inputs[397] -0.017285*inputs[398] +0.0125288*inputs[399] +0.00201011*inputs[400] -0.0292779*inputs[401] +0.0116808*inputs[402] -0.0251483*inputs[403] +0.00535756*inputs[404] -0.000287656*inputs[405] +0.0652932*inputs[406] +0.000430832*inputs[407] +0.0295221*inputs[408] +0.0132182*inputs[409] -0.00316316*inputs[410] +0.0468948*inputs[411] -0.00414546*inputs[412] +0.00330387*inputs[413] +0.0208395*inputs[414] -0.00292295*inputs[415] +0.024772*inputs[416] -0.000489914*inputs[417] +0.0180193*inputs[418] +0.0108353*inputs[419] +0.000820726*inputs[420] +0.0314013*inputs[421] +0.0033039*inputs[422] +0.0277288*inputs[423] +0.0141486*inputs[424] +0.0193386*inputs[425] +0.0122605*inputs[426] +0.037501*inputs[427] -0.0116167*inputs[428] -0.00862762*inputs[429] -0.00917346*inputs[430] -0.011479*inputs[431] +0.047738*inputs[432] -0.0109328*inputs[433] +0.0195921*inputs[434] +0.00986176*inputs[435] +0.00518847*inputs[436] -0.00402996*inputs[437] -0.000590079*inputs[438] +0.0348751*inputs[439] +0.00368237*inputs[440] +0.00330387*inputs[441] +0.00999964*inputs[442] +0.016214*inputs[443] +0.00689093*inputs[444] +0.0323292*inputs[445] +0.077339*inputs[446] -0.0251869*inputs[447] +0.0183571*inputs[448] +0.0114474*inputs[449] +0.00269617*inputs[450] +0.0401633*inputs[451] +0.0129577*inputs[452] +0.00971623*inputs[453] -0.0158128*inputs[454] +0.02528*inputs[455] +0.0252801*inputs[456] -0.0047364*inputs[457] -0.00481041*inputs[458] -0.0556848*inputs[459] +0.00269616*inputs[460] +0.0300771*inputs[461] +0.0229326*inputs[462] +0.00190539*inputs[463] +0.0225546*inputs[464] +0.00269618*inputs[465] -0.0182966*inputs[466] +0.0674162*inputs[467] -0.0072236*inputs[468] -0.00879913*inputs[469] -0.00121671*inputs[470] -0.000364371*inputs[471] -0.0381658*inputs[472] +0.0131831*inputs[473] -0.0113687*inputs[474] -0.00854009*inputs[475] +0.0146995*inputs[476] +0.00943761*inputs[477] +0.0931931*inputs[478] +0.0283824*inputs[479] +0.0224424*inputs[480] -0.0231086*inputs[481] +0.00424683*inputs[482] +0.0124936*inputs[483] +0.00269617*inputs[484] +0.00366209*inputs[485] +0.0097914*inputs[486] +0.00269616*inputs[487] -0.0263653*inputs[488] +0.0133447*inputs[489] +0.0818134*inputs[490] +0.00268466*inputs[491] -0.00631113*inputs[492] +0.0226763*inputs[493] -0.00258554*inputs[494] -0.00474296*inputs[495] +0.0264158*inputs[496] +0.0312335*inputs[497] +0.0193958*inputs[498] -0.00341677*inputs[499] -0.041064*inputs[500] -0.0200887*inputs[501] +0.0187599*inputs[502] +0.00269617*inputs[503] +0.0301588*inputs[504] +0.00269616*inputs[505] +0.00166415*inputs[506] -0.0304069*inputs[507] -0.00187187*inputs[508] +0.0247592*inputs[509] -0.0219272*inputs[510] -0.0127819*inputs[511] +0.0411048*inputs[512] +0.0155872*inputs[513] +0.00269616*inputs[514] +0.041808*inputs[515] +0.0186835*inputs[516] +0.0026962*inputs[517] +0.00322455*inputs[518] +0.0119906*inputs[519] +0.0223453*inputs[520] +0.010748*inputs[521] +0.0105193*inputs[522] +0.0105193*inputs[523] +0.0412583*inputs[524] -0.0221841*inputs[525] +0.0138264*inputs[526] +0.0119632*inputs[527] -0.00269698*inputs[528] +0.00269616*inputs[529] +0.0358483*inputs[530] -0.00823534*inputs[531] -0.00823546*inputs[532] +0.00269618*inputs[533] -0.0135525*inputs[534] -0.00795925*inputs[535] +0.0139973*inputs[536] +0.00599004*inputs[537] +0.00923268*inputs[538] -0.0131853*inputs[539] -0.0131853*inputs[540] +0.00275506*inputs[541] +0.00329726*inputs[542] -0.00141509*inputs[543] +0.0409769*inputs[544] +0.00269616*inputs[545] +0.00269617*inputs[546] -0.00490029*inputs[547] +0.0145222*inputs[548] +0.00269618*inputs[549] +0.00269617*inputs[550] +0.0353506*inputs[551] +0.0206225*inputs[552] -0.0079593*inputs[553] +0.00445681*inputs[554] +0.00269616*inputs[555] -0.00580465*inputs[556] +0.0341714*inputs[557] +0.00399775*inputs[558] +0.046932*inputs[559] +0.00299012*inputs[560] -0.0258466*inputs[561] +0.000481938*inputs[562] +0.0395921*inputs[563] -0.0400005*inputs[564] -0.0122929*inputs[565] +0.00472476*inputs[566] +0.0241621*inputs[567] +0.00269617*inputs[568] -0.00931177*inputs[569] -0.00540253*inputs[570] +0.0278519*inputs[571] +0.013371*inputs[572] -0.00531351*inputs[573] -0.00966976*inputs[574] +0.0171118*inputs[575] -0.00312392*inputs[576] +0.00268464*inputs[577] +0.00269616*inputs[578] +0.0237085*inputs[579] -0.0193709*inputs[580] +0.00190543*inputs[581] +0.00785688*inputs[582] +0.0152257*inputs[583] -0.000321288*inputs[584] +0.0282459*inputs[585] -0.00384064*inputs[586] -0.0148664*inputs[587] +0.00269616*inputs[588] +0.00269616*inputs[589] +0.025058*inputs[590] -0.00187193*inputs[591] +0.0310437*inputs[592] +0.00468493*inputs[593] +0.0113792*inputs[594] -0.00722455*inputs[595] -0.0228254*inputs[596] -0.000759267*inputs[597] +0.00426279*inputs[598] -0.00942006*inputs[599] -0.00913753*inputs[600] +0.00225667*inputs[601] +0.00269616*inputs[602] -0.0247607*inputs[603] -0.0204866*inputs[604] +0.0266481*inputs[605] +0.0310085*inputs[606] +0.00269617*inputs[607] +0.0404291*inputs[608] +0.0126529*inputs[609] -0.00743405*inputs[610] -0.0162651*inputs[611] +0.00484769*inputs[612] -0.00146771*inputs[613] +0.00269616*inputs[614] +0.00269616*inputs[615] +0.047324*inputs[616] -0.01572*inputs[617] +0.00845*inputs[618] +0.00472484*inputs[619] -0.000777222*inputs[620] +0.000236434*inputs[621] +0.0120807*inputs[622] -0.00250139*inputs[623] +0.0172184*inputs[624] -0.012546*inputs[625] +0.00269616*inputs[626] +0.0130526*inputs[627] -0.0148664*inputs[628] +0.0237596*inputs[629] +0.0535475*inputs[630] -0.0271516*inputs[631] +0.00322456*inputs[632] +0.00223271*inputs[633] -0.00116601*inputs[634] -0.00522035*inputs[635] -0.001166*inputs[636] -0.0256674*inputs[637] -0.023222*inputs[638] -0.0131814*inputs[639] -0.0156963*inputs[640] +0.017987*inputs[641] +0.00521893*inputs[642] +0.00269621*inputs[643] +0.00422997*inputs[644] +0.00846001*inputs[645] +0.0129577*inputs[646] -0.000820834*inputs[647] +0.0019054*inputs[648] +0.00401467*inputs[649] +0.00540733*inputs[650] +0.0118282*inputs[651] +0.0118282*inputs[652] +0.0118282*inputs[653] -0.00579825*inputs[654] +0.00190541*inputs[655] +0.00190539*inputs[656] +0.0019054*inputs[657] +0.00190539*inputs[658] +0.00190539*inputs[659] +0.00190544*inputs[660] +0.00190539*inputs[661] +0.00882027*inputs[662] +0.0111846*inputs[663] +0.0111846*inputs[664] +0.00190539*inputs[665] -0.0120896*inputs[666] +0.0187522*inputs[667] +0.0187522*inputs[668] +0.0309024*inputs[669] -0.0128822*inputs[670] +0.0147281*inputs[671] +0.00190539*inputs[672] -0.00461786*inputs[673] -0.00461786*inputs[674] -0.00461783*inputs[675] +0.0119346*inputs[676] +0.0119346*inputs[677] +0.0119346*inputs[678] +0.0235503*inputs[679] -0.0230601*inputs[680] -0.0230601*inputs[681] -0.0230601*inputs[682] +0.00190539*inputs[683] +0.00190539*inputs[684] +0.00190543*inputs[685] +0.00190546*inputs[686] +0.00190539*inputs[687] +0.00944482*inputs[688] +0.00944482*inputs[689] +0.00190543*inputs[690] +0.00190542*inputs[691] +0.000446682*inputs[692] +0.000446705*inputs[693] +0.000446654*inputs[694] +0.000446664*inputs[695] +0.00190539*inputs[696] +0.00190539*inputs[697] +0.00190539*inputs[698] -0.0022265*inputs[699] -0.0022265*inputs[700] +0.0129634*inputs[701] +0.00461286*inputs[702] +0.00461286*inputs[703] +0.00461286*inputs[704] +0.00461286*inputs[705] -0.00607193*inputs[706] +0.00190544*inputs[707] +0.00190539*inputs[708] +0.00190544*inputs[709] +0.00190539*inputs[710] +0.00190543*inputs[711] +0.00190542*inputs[712] +0.0641734*inputs[713] +0.0235633*inputs[714] +0.0448698*inputs[715] -0.00730628*inputs[716] -0.00730628*inputs[717] +0.00190539*inputs[718] +0.00190539*inputs[719] +0.0108538*inputs[720] +0.0108538*inputs[721] -0.00122442*inputs[722] +0.00279319*inputs[723] +0.00190539*inputs[724] +0.00190539*inputs[725] +0.00190539*inputs[726] +0.00190539*inputs[727] +0.00190539*inputs[728] +0.0412982*inputs[729] -0.00312397*inputs[730] -0.00312397*inputs[731] -0.00312397*inputs[732] -0.00312397*inputs[733] -0.0031239*inputs[734] +0.00190548*inputs[735] +0.00190539*inputs[736] +0.00190539*inputs[737] -0.0164246*inputs[738] +0.00190543*inputs[739] +0.00190547*inputs[740] +0.00190541*inputs[741] -0.0148738*inputs[742] -0.0148738*inputs[743] -0.00485895*inputs[744] -0.0286594*inputs[745] +0.000240095*inputs[746] +0.000240098*inputs[747] +0.0138196*inputs[748] +0.0138196*inputs[749] -0.0066342*inputs[750] -0.00663418*inputs[751] -0.0124138*inputs[752] +0.00190539*inputs[753] +0.00190539*inputs[754] -0.00483238*inputs[755] -0.00483238*inputs[756] -0.00483238*inputs[757] -0.00483238*inputs[758] -0.00964396*inputs[759] -0.00964398*inputs[760] +0.00190542*inputs[761] +0.00814825*inputs[762] +0.00814828*inputs[763] +0.00814829*inputs[764] +0.00814825*inputs[765] +0.00814826*inputs[766] +0.00814825*inputs[767] +0.0132725*inputs[768] +0.0132725*inputs[769] +0.00190543*inputs[770] -0.0120389*inputs[771] -0.0120389*inputs[772] +0.00190539*inputs[773] +0.00190544*inputs[774] +0.00190539*inputs[775] +0.00190542*inputs[776] +0.00190544*inputs[777] +0.00190539*inputs[778] +0.00190539*inputs[779] +0.00190541*inputs[780] +0.0448244*inputs[781] +0.0237263*inputs[782] +0.0237263*inputs[783] +0.00865729*inputs[784] +0.00865729*inputs[785] +0.0216407*inputs[786] +0.0019054*inputs[787] +0.00190539*inputs[788] +0.00190539*inputs[789] -0.0147352*inputs[790] +0.00190539*inputs[791] -8.39026e-05*inputs[792] +0.00190539*inputs[793] +0.00190539*inputs[794] -0.0150568*inputs[795] -0.0150568*inputs[796] -0.0228123*inputs[797] +0.00190545*inputs[798] +0.00190542*inputs[799] +0.00190539*inputs[800] -0.00182959*inputs[801] -0.00182958*inputs[802] +0.0131553*inputs[803] +0.00190539*inputs[804] +0.00673431*inputs[805] +0.00190539*inputs[806] -0.00724701*inputs[807] +0.00190549*inputs[808] +0.00190539*inputs[809] +0.00190539*inputs[810] +0.00190539*inputs[811] +0.0067343*inputs[812] -0.0409123*inputs[813] -0.00258753*inputs[814] -0.0025875*inputs[815] +0.00190539*inputs[816] -0.00724699*inputs[817] +0.00190539*inputs[818] +0.00190547*inputs[819] +0.00190541*inputs[820] +0.00190539*inputs[821] +0.0328228*inputs[822] +0.0284956*inputs[823] +0.00472992*inputs[824] +0.00190539*inputs[825] +0.00190539*inputs[826] -0.007247*inputs[827] +0.00472992*inputs[828] -0.0317628*inputs[829] -0.0131562*inputs[830] -0.00673523*inputs[831] -0.00673529*inputs[832] -0.0127372*inputs[833] +0.00472992*inputs[834] -0.00673528*inputs[835] -0.00674744*inputs[836] -0.0101106*inputs[837] +0.00190545*inputs[838] -0.0136873*inputs[839] +0.00190539*inputs[840] +0.000411259*inputs[841] -0.0195109*inputs[842] +0.00951319*inputs[843] +0.00951317*inputs[844] +0.00951317*inputs[845] +0.00951317*inputs[846] +0.00951319*inputs[847] -0.0239119*inputs[848] +0.014226*inputs[849] +0.014226*inputs[850] -0.0239119*inputs[851] +0.0242395*inputs[852] +0.0132867*inputs[853] -0.0134577*inputs[854] -0.00890025*inputs[855] +0.00190539*inputs[856] +0.00190539*inputs[857] +0.00190539*inputs[858] -0.00239122*inputs[859] +0.00190539*inputs[860] +0.00190539*inputs[861] +0.0019054*inputs[862] +0.00190539*inputs[863] +0.00190539*inputs[864] -0.023507*inputs[865] +0.00374521*inputs[866] +0.0132867*inputs[867] +0.0162372*inputs[868] +0.00190541*inputs[869] +0.00850746*inputs[870] +0.00850746*inputs[871] +0.00190546*inputs[872] +0.00776101*inputs[873] +0.00776102*inputs[874] +0.00776101*inputs[875] +0.00190539*inputs[876] -0.00674744*inputs[877] +0.0226056*inputs[878] +0.0245508*inputs[879] +0.0226056*inputs[880] -0.00810347*inputs[881] +0.0371551*inputs[882] +0.00190539*inputs[883] +0.00190544*inputs[884] -0.00330074*inputs[885] -0.00330066*inputs[886] +0.00190539*inputs[887] +0.0019054*inputs[888] +0.0019054*inputs[889] +0.00190544*inputs[890] +0.00190546*inputs[891] +0.00190541*inputs[892] +0.00190539*inputs[893] +0.00190539*inputs[894] +0.00190539*inputs[895] +0.00190539*inputs[896] +0.00729153*inputs[897] +0.0019054*inputs[898] +0.00182103*inputs[899] +0.00729147*inputs[900] -0.00674744*inputs[901] -0.00786339*inputs[902] +0.0194157*inputs[903] +0.00190539*inputs[904] -0.00786339*inputs[905] -0.00786339*inputs[906] +0.00130806*inputs[907] +0.00130813*inputs[908] +0.00190539*inputs[909] +0.00190539*inputs[910] +0.00190542*inputs[911] -0.0313609*inputs[912] +0.0224326*inputs[913] +0.00190542*inputs[914] +0.00190539*inputs[915] +0.0194158*inputs[916] +0.00169608*inputs[917] +0.00190539*inputs[918] +0.00169603*inputs[919] +0.00169604*inputs[920] +0.0194157*inputs[921] +0.0380202*inputs[922] -0.0134712*inputs[923] +0.0129149*inputs[924] +0.00190539*inputs[925] +0.00190544*inputs[926] +0.00190539*inputs[927] +0.0019054*inputs[928] +0.0332781*inputs[929] +0.00190539*inputs[930] +0.0019054*inputs[931] +0.0310019*inputs[932] +0.0124936*inputs[933] -0.00741603*inputs[934] +0.0019054*inputs[935] +0.0124936*inputs[936] +0.00845327*inputs[937] -0.00741603*inputs[938] +0.0163944*inputs[939] +0.00409729*inputs[940] +0.00409731*inputs[941] +0.00409733*inputs[942] +0.00409732*inputs[943] -0.000875719*inputs[944] +0.0187526*inputs[945] -0.000875719*inputs[946] -0.0242571*inputs[947] +0.00190547*inputs[948] -0.0073344*inputs[949] -0.00733437*inputs[950] +0.0187526*inputs[951] +0.0216608*inputs[952] +0.0019054*inputs[953] +0.0019054*inputs[954] -0.0109842*inputs[955] -0.0109842*inputs[956] -0.0109842*inputs[957] -0.0109841*inputs[958] +0.0197507*inputs[959] +0.00190539*inputs[960] +0.00190539*inputs[961] -0.0101292*inputs[962] -0.0101292*inputs[963] -0.00350614*inputs[964] -0.0295693*inputs[965] +0.00190539*inputs[966] +0.00190539*inputs[967] +0.00190539*inputs[968] +0.0328232*inputs[969] +0.00190548*inputs[970] -0.0188084*inputs[971] +0.0019054*inputs[972] +0.00190541*inputs[973] +0.00190549*inputs[974] +0.00190541*inputs[975] +0.051228*inputs[976] +0.0188752*inputs[977] -0.0133708*inputs[978] +0.00190539*inputs[979] +0.0206702*inputs[980] +0.00190539*inputs[981] -0.0267356*inputs[982] -0.0267356*inputs[983] +0.00190543*inputs[984] +0.00190539*inputs[985] +0.00190539*inputs[986] -4.87343e-05*inputs[987] +0.00190548*inputs[988] +0.00190539*inputs[989] +0.00190539*inputs[990] +0.00796236*inputs[991] +0.0165107*inputs[992] +0.00190539*inputs[993] -0.00380751*inputs[994] +0.00190539*inputs[995] +0.0201342*inputs[996] -0.0309597*inputs[997] +0.00796238*inputs[998] +0.0019054*inputs[999] +0.00190543*inputs[1000] +0.00190539*inputs[1001] +0.0108876*inputs[1002] +0.00411988*inputs[1003] +0.00796237*inputs[1004] +0.00190546*inputs[1005] +0.00190539*inputs[1006] +0.00190544*inputs[1007] +0.0239486*inputs[1008] -0.0121176*inputs[1009] -0.0121176*inputs[1010] -0.0121176*inputs[1011] +0.00190547*inputs[1012] +0.00190539*inputs[1013] +0.00190539*inputs[1014] +0.00190545*inputs[1015] +0.0105541*inputs[1016] +0.0105541*inputs[1017] -0.015221*inputs[1018] -0.015221*inputs[1019] +0.0255105*inputs[1020] +0.00128423*inputs[1021] +0.00190539*inputs[1022] +0.00190539*inputs[1023] +0.00796236*inputs[1024] +0.0038415*inputs[1025] +0.0038415*inputs[1026] +0.00384153*inputs[1027] +0.0019054*inputs[1028] +0.00190541*inputs[1029] +0.00190539*inputs[1030] +0.00796237*inputs[1031] -0.00571791*inputs[1032] -0.0137037*inputs[1033] -0.00571792*inputs[1034] +0.0163238*inputs[1035] +0.0163238*inputs[1036] +0.0163238*inputs[1037] +0.0163238*inputs[1038] +0.0163238*inputs[1039] +0.00190539*inputs[1040] +0.00190543*inputs[1041] +0.0019054*inputs[1042] -0.0110299*inputs[1043] +0.0172569*inputs[1044] +0.0172569*inputs[1045] +0.0172569*inputs[1046] -0.0110299*inputs[1047] -0.0103*inputs[1048] +0.00190542*inputs[1049] +0.0172358*inputs[1050] -0.0103*inputs[1051] -0.0110299*inputs[1052] +0.00190539*inputs[1053] -0.00398024*inputs[1054] -0.00398024*inputs[1055] -0.00398023*inputs[1056] -0.00398024*inputs[1057] +0.0150431*inputs[1058] -0.032872*inputs[1059] +0.00190539*inputs[1060] +0.00190539*inputs[1061] +0.00190542*inputs[1062] -0.0150679*inputs[1063] -0.0331768*inputs[1064] +0.00190539*inputs[1065] +0.0111447*inputs[1066] +0.0111447*inputs[1067] +0.00190542*inputs[1068] +0.00190541*inputs[1069] +0.0019054*inputs[1070] +0.00190539*inputs[1071] +0.00190543*inputs[1072] +0.00190539*inputs[1073] +0.00190539*inputs[1074] +0.00190539*inputs[1075] +0.00190539*inputs[1076] +0.00190539*inputs[1077] -0.00508041*inputs[1078] -0.0241258*inputs[1079] -0.0181901*inputs[1080] -0.00508042*inputs[1081] -0.00508042*inputs[1082] +0.00477285*inputs[1083] +0.00477284*inputs[1084] +0.00477284*inputs[1085] +0.00190548*inputs[1086] +0.00190546*inputs[1087] -0.00508039*inputs[1088] +0.0400756*inputs[1089] +0.0019054*inputs[1090] -0.0103*inputs[1091] +0.0222983*inputs[1092] +0.00190539*inputs[1093] -0.0101203*inputs[1094] +0.00190539*inputs[1095] -0.0103*inputs[1096] +0.00190539*inputs[1097] +0.00190539*inputs[1098] -3.48853e-05*inputs[1099] -3.4862e-05*inputs[1100] +0.00190539*inputs[1101] +0.00190539*inputs[1102] -0.0128527*inputs[1103] -0.0212513*inputs[1104] +0.00190549*inputs[1105] +0.00190542*inputs[1106] +0.0019054*inputs[1107] +0.00190541*inputs[1108] +0.0148898*inputs[1109] -0.00390589*inputs[1110] -0.029231*inputs[1111] +0.0165441*inputs[1112] +0.00190542*inputs[1113] +0.0153063*inputs[1114] +0.00190541*inputs[1115] +0.0601566*inputs[1116] +0.0301344*inputs[1117] -0.0152477*inputs[1118] +0.00190539*inputs[1119] +0.00190539*inputs[1120] +0.00190543*inputs[1121] +0.00190544*inputs[1122] +0.00190539*inputs[1123] +0.00190545*inputs[1124] +0.00190539*inputs[1125] +0.00618964*inputs[1126] +0.00618965*inputs[1127] +0.00190542*inputs[1128] +0.00190543*inputs[1129] +0.0267135*inputs[1130] +0.037463*inputs[1131] -0.0268655*inputs[1132] +0.00855967*inputs[1133] +0.00855964*inputs[1134] +0.0112109*inputs[1135] +0.0112109*inputs[1136] +0.0112109*inputs[1137] +0.0153636*inputs[1138] +0.00190539*inputs[1139] +0.00190544*inputs[1140] -0.00556043*inputs[1141] +0.0153063*inputs[1142] +0.00265233*inputs[1143] +0.00265226*inputs[1144] +0.00265226*inputs[1145] +0.00265228*inputs[1146] -0.0265725*inputs[1147] -0.0265725*inputs[1148] -0.0113089*inputs[1149] +0.00190539*inputs[1150] +0.00190539*inputs[1151] +0.0019054*inputs[1152] -0.00713571*inputs[1153] -0.00713571*inputs[1154] -0.0071357*inputs[1155] +0.0149274*inputs[1156] +0.0215282*inputs[1157] +0.032593*inputs[1158] +0.00190544*inputs[1159] +0.0019054*inputs[1160] -0.00539418*inputs[1161] +0.00190541*inputs[1162] -0.0225041*inputs[1163] +0.00190539*inputs[1164] +0.0019054*inputs[1165] +0.0019054*inputs[1166] +0.0241817*inputs[1167] +0.00190541*inputs[1168] +0.00190539*inputs[1169] +0.00190539*inputs[1170] +0.00190539*inputs[1171] +0.00190539*inputs[1172] -0.00355374*inputs[1173] -0.0108333*inputs[1174] -0.0108333*inputs[1175] -0.0108333*inputs[1176] -0.0108333*inputs[1177] -0.0121627*inputs[1178] -0.0121627*inputs[1179] -0.0121627*inputs[1180] -0.0121627*inputs[1181] +0.00190539*inputs[1182] +0.00190542*inputs[1183] +0.00190539*inputs[1184] +0.00190539*inputs[1185] +0.0338501*inputs[1186] +0.0019054*inputs[1187] +0.00190547*inputs[1188] +0.00190544*inputs[1189] -0.0287619*inputs[1190] +0.0019054*inputs[1191] -0.00300435*inputs[1192] +0.0328276*inputs[1193] -0.0268368*inputs[1194] +0.00190539*inputs[1195] +0.00190539*inputs[1196] -0.0291479*inputs[1197] -0.0205376*inputs[1198] -0.0205376*inputs[1199] +0.0240619*inputs[1200] +0.00190542*inputs[1201] +0.00190543*inputs[1202] +0.00190539*inputs[1203] +0.00190539*inputs[1204] +0.0227893*inputs[1205] -0.00781438*inputs[1206] -0.00781438*inputs[1207] +0.00190539*inputs[1208] +0.00190539*inputs[1209] +0.00190543*inputs[1210] +0.00190539*inputs[1211] +0.00190548*inputs[1212] +0.00190539*inputs[1213] +0.00190541*inputs[1214] +0.00190539*inputs[1215] +0.00190539*inputs[1216] +0.00190543*inputs[1217] +0.0019054*inputs[1218] -0.0199041*inputs[1219] -0.0199041*inputs[1220] -0.0199041*inputs[1221] +0.0126067*inputs[1222] +0.0126067*inputs[1223] +0.0126067*inputs[1224] +0.019548*inputs[1225] +0.019548*inputs[1226] +0.0019054*inputs[1227] +0.00190547*inputs[1228] +0.00190539*inputs[1229] +0.00592014*inputs[1230] +0.00592013*inputs[1231] +0.0328231*inputs[1232] +0.00190539*inputs[1233] +0.0019054*inputs[1234] +0.00190544*inputs[1235] -0.0304286*inputs[1236] +0.00190544*inputs[1237] +0.0186216*inputs[1238] +0.0182872*inputs[1239] +0.0182872*inputs[1240] +0.00190539*inputs[1241] +0.00190543*inputs[1242] +0.00190539*inputs[1243] +0.00190539*inputs[1244] +0.00190539*inputs[1245] +0.00768596*inputs[1246] +0.00768596*inputs[1247] +0.00768598*inputs[1248] -0.0134482*inputs[1249] +0.00190539*inputs[1250] +0.0019054*inputs[1251] +0.00741134*inputs[1252] +0.00190546*inputs[1253] +0.0159793*inputs[1254] +0.00471656*inputs[1255] +0.00471656*inputs[1256] +0.0180172*inputs[1257] +0.0100527*inputs[1258] +0.0100527*inputs[1259] -0.00306588*inputs[1260] -0.00413581*inputs[1261] -0.00306589*inputs[1262] -0.00306591*inputs[1263] -0.00306591*inputs[1264] +0.0326089*inputs[1265] -0.0391757*inputs[1266] +0.00190539*inputs[1267] +0.00190539*inputs[1268] +0.00190543*inputs[1269] -0.021503*inputs[1270] -0.0205987*inputs[1271] +0.00717107*inputs[1272] +0.0071711*inputs[1273] -0.010719*inputs[1274] -0.010719*inputs[1275] +0.00190541*inputs[1276] +0.00190541*inputs[1277] -0.0213592*inputs[1278] +0.00190539*inputs[1279] +0.0019054*inputs[1280] -0.00413583*inputs[1281] +0.00190539*inputs[1282] +0.00190539*inputs[1283] +0.0299753*inputs[1284] +0.0132546*inputs[1285] +0.0475916*inputs[1286] +0.00190541*inputs[1287] +0.0019054*inputs[1288] +0.0019054*inputs[1289] +0.00545894*inputs[1290] +0.0115681*inputs[1291] +0.0164102*inputs[1292] +0.0302495*inputs[1293] +0.012611*inputs[1294] +0.00674929*inputs[1295] +0.00190542*inputs[1296] +0.0115681*inputs[1297] -0.0310604*inputs[1298] +0.0019054*inputs[1299] +0.00190539*inputs[1300] +0.00190544*inputs[1301] +0.00190541*inputs[1302] -0.00481038*inputs[1303] +0.0019054*inputs[1304] -0.0105773*inputs[1305] -0.0105773*inputs[1306] -0.0105772*inputs[1307] +0.00190541*inputs[1308] -0.00860976*inputs[1309] +0.00190541*inputs[1310] +0.00190539*inputs[1311] +0.00190541*inputs[1312] +0.00702307*inputs[1313] +0.00702307*inputs[1314] +0.00190539*inputs[1315] +0.00190541*inputs[1316] +0.00554941*inputs[1317] +0.00190543*inputs[1318] +0.0166917*inputs[1319] -0.0316661*inputs[1320] +0.00190542*inputs[1321] +0.00658457*inputs[1322] +0.00190541*inputs[1323] +0.00190539*inputs[1324] -0.0391731*inputs[1325] -0.00883225*inputs[1326] +0.0019054*inputs[1327] +0.00190539*inputs[1328] +0.00190539*inputs[1329] +0.00190539*inputs[1330] +0.0142752*inputs[1331] +0.00190549*inputs[1332] +0.0019054*inputs[1333] +0.00190547*inputs[1334] +0.00190539*inputs[1335] -0.0129045*inputs[1336] -0.0129045*inputs[1337] -0.0129045*inputs[1338] +0.0114842*inputs[1339] +0.0114842*inputs[1340] +0.00190542*inputs[1341] +0.00811074*inputs[1342] +0.00190539*inputs[1343] +0.00190539*inputs[1344] +0.0019054*inputs[1345] +0.0169943*inputs[1346] +0.00190539*inputs[1347] +0.00190539*inputs[1348] +0.0019054*inputs[1349] +0.00190542*inputs[1350] +0.00190545*inputs[1351] +0.00190546*inputs[1352] -0.0121163*inputs[1353] -0.0121162*inputs[1354] +0.00190539*inputs[1355] +0.00190539*inputs[1356] -0.0331768*inputs[1357] -0.00242069*inputs[1358] +0.00190539*inputs[1359] +0.00190541*inputs[1360] +0.00190539*inputs[1361] +0.00190539*inputs[1362] +0.00190539*inputs[1363] -0.0222646*inputs[1364] -0.0222646*inputs[1365] +0.000151219*inputs[1366] +0.00015122*inputs[1367] +0.000151241*inputs[1368] +0.000151222*inputs[1369] +0.000151219*inputs[1370] +0.0220701*inputs[1371] +0.00653657*inputs[1372] +0.00653661*inputs[1373] -0.00773264*inputs[1374] +0.00665487*inputs[1375] +0.00665485*inputs[1376] +0.0142584*inputs[1377] +0.0505985*inputs[1378] -0.00152552*inputs[1379] +0.00190539*inputs[1380] +0.00190539*inputs[1381] +0.00190547*inputs[1382] +0.00190539*inputs[1383] +0.0347539*inputs[1384] +0.0164765*inputs[1385] +0.0164765*inputs[1386] +0.00190544*inputs[1387] +0.00190539*inputs[1388] +0.0019054*inputs[1389] +0.030306*inputs[1390] +0.00190539*inputs[1391] +0.00190539*inputs[1392] +0.00190549*inputs[1393] +0.00190544*inputs[1394] +0.0019054*inputs[1395] 
		combinations[1] = -0.0445378 -0.0847353*inputs[0] -0.0471912*inputs[1] -0.225775*inputs[2] -0.156737*inputs[3] -0.157949*inputs[4] -0.0827475*inputs[5] -0.0161041*inputs[6] -0.0333671*inputs[7] -0.0510973*inputs[8] -0.00548516*inputs[9] -0.0117511*inputs[10] -0.0893942*inputs[11] -0.0355982*inputs[12] -0.103443*inputs[13] -0.184019*inputs[14] -0.035059*inputs[15] -0.00958436*inputs[16] -0.0201421*inputs[17] -0.112632*inputs[18] +0.107123*inputs[19] -0.00690704*inputs[20] -0.0168266*inputs[21] -0.028528*inputs[22] -0.00431353*inputs[23] +0.00138911*inputs[24] +0.0201467*inputs[25] -0.129393*inputs[26] +0.0144867*inputs[27] -0.0101167*inputs[28] -0.00132185*inputs[29] -0.0181941*inputs[30] +0.0208967*inputs[31] +0.0949329*inputs[32] -0.107319*inputs[33] +0.0807242*inputs[34] -0.107588*inputs[35] -0.0931483*inputs[36] -0.00860369*inputs[37] +0.00703272*inputs[38] +0.127869*inputs[39] -0.0415331*inputs[40] +0.188294*inputs[41] -0.0162703*inputs[42] +0.0452256*inputs[43] -0.0363233*inputs[44] -0.0130849*inputs[45] +0.018324*inputs[46] +0.0647335*inputs[47] +0.0810638*inputs[48] -0.0947214*inputs[49] -0.0030498*inputs[50] -0.00334102*inputs[51] -0.00261113*inputs[52] -0.014066*inputs[53] +0.00169844*inputs[54] +0.0586033*inputs[55] -0.0641108*inputs[56] -0.0365415*inputs[57] +0.110851*inputs[58] +0.00279687*inputs[59] +0.132162*inputs[60] +0.0293188*inputs[61] +0.0296123*inputs[62] +0.0345088*inputs[63] -0.00176787*inputs[64] +0.155046*inputs[65] +0.0825484*inputs[66] -0.0473098*inputs[67] +0.0425301*inputs[68] -0.000256828*inputs[69] -0.0125954*inputs[70] +0.00268501*inputs[71] -0.011664*inputs[72] +0.0121983*inputs[73] +0.0921066*inputs[74] -0.00903488*inputs[75] -0.103975*inputs[76] -0.0214301*inputs[77] -0.0191075*inputs[78] -0.0131605*inputs[79] +0.0303853*inputs[80] +0.0394864*inputs[81] -0.0115625*inputs[82] -0.0934033*inputs[83] -0.0228211*inputs[84] +0.0435127*inputs[85] -0.0447603*inputs[86] +0.0170929*inputs[87] +0.0253661*inputs[88] +0.0147146*inputs[89] +0.00491171*inputs[90] +0.0239148*inputs[91] -0.00157425*inputs[92] +0.0231703*inputs[93] +0.104226*inputs[94] +0.0689444*inputs[95] -0.00944092*inputs[96] -0.0252946*inputs[97] +0.0121799*inputs[98] -0.0375569*inputs[99] -0.00570972*inputs[100] +0.0297536*inputs[101] +0.0213163*inputs[102] +0.0128931*inputs[103] +0.0473892*inputs[104] -0.00375925*inputs[105] +0.137614*inputs[106] -0.0045038*inputs[107] +0.0837952*inputs[108] -0.00650885*inputs[109] -0.0479349*inputs[110] -0.0629375*inputs[111] +0.0194525*inputs[112] -0.0216859*inputs[113] +0.0631641*inputs[114] -0.0197618*inputs[115] +0.00264454*inputs[116] +0.0162972*inputs[117] +0.0324477*inputs[118] +0.0175617*inputs[119] +0.0304775*inputs[120] -0.0835911*inputs[121] -0.0778085*inputs[122] +0.0811802*inputs[123] +0.0284311*inputs[124] +0.0151315*inputs[125] +0.00397971*inputs[126] +0.00317815*inputs[127] -0.00702579*inputs[128] -0.0047785*inputs[129] -0.00909572*inputs[130] -0.148734*inputs[131] -0.0218995*inputs[132] -0.0115887*inputs[133] +0.0202773*inputs[134] +0.016402*inputs[135] +0.0832903*inputs[136] +0.0068862*inputs[137] +0.0216375*inputs[138] +0.0847752*inputs[139] -0.0109081*inputs[140] -0.0217936*inputs[141] -0.0294631*inputs[142] -0.0419255*inputs[143] +0.0132618*inputs[144] -0.00722105*inputs[145] -0.082902*inputs[146] +0.0212839*inputs[147] +0.025059*inputs[148] +0.0378146*inputs[149] +0.0203931*inputs[150] +0.0142318*inputs[151] +0.0327232*inputs[152] +0.000546062*inputs[153] +0.062367*inputs[154] +0.0167815*inputs[155] -0.00265287*inputs[156] +0.00929135*inputs[157] +0.0256024*inputs[158] -0.0649198*inputs[159] +0.00319683*inputs[160] +0.0140867*inputs[161] -0.00427045*inputs[162] +0.0233903*inputs[163] +0.00778407*inputs[164] +0.00152605*inputs[165] -0.0213452*inputs[166] -0.00746453*inputs[167] +0.000346064*inputs[168] +0.0235048*inputs[169] +0.0125664*inputs[170] +0.0373023*inputs[171] -0.00682867*inputs[172] -0.0228292*inputs[173] +0.0184272*inputs[174] -0.0599874*inputs[175] -0.0121458*inputs[176] -0.00164318*inputs[177] +0.0705405*inputs[178] -0.047243*inputs[179] +0.0570612*inputs[180] -0.00217852*inputs[181] -0.00301289*inputs[182] -0.0110746*inputs[183] +0.0111379*inputs[184] +0.00135506*inputs[185] +0.0118006*inputs[186] +0.0119523*inputs[187] -0.0159997*inputs[188] +0.00867874*inputs[189] +0.0255963*inputs[190] -0.00768424*inputs[191] -0.0359685*inputs[192] +0.00336761*inputs[193] -0.049384*inputs[194] -0.0198621*inputs[195] -0.0206451*inputs[196] +0.0369214*inputs[197] -0.0350179*inputs[198] +0.006063*inputs[199] +0.00340965*inputs[200] -0.00455623*inputs[201] +0.00384556*inputs[202] +0.0130113*inputs[203] -0.0279722*inputs[204] +0.0406423*inputs[205] -0.0231106*inputs[206] +0.0268943*inputs[207] +0.0176391*inputs[208] +0.0539333*inputs[209] -0.00108851*inputs[210] +0.00520273*inputs[211] +0.0279205*inputs[212] +0.00320581*inputs[213] -0.0164226*inputs[214] -0.0557185*inputs[215] -0.0275531*inputs[216] +0.0219031*inputs[217] -0.0149706*inputs[218] +0.00730957*inputs[219] -0.0169914*inputs[220] +0.0188783*inputs[221] -0.0140749*inputs[222] -0.00291009*inputs[223] +0.0390736*inputs[224] +0.0537295*inputs[225] -0.000728139*inputs[226] +0.00733852*inputs[227] -0.00936288*inputs[228] -0.0162254*inputs[229] -0.0199657*inputs[230] -0.0049075*inputs[231] -0.0197248*inputs[232] -0.01016*inputs[233] -0.00179478*inputs[234] -0.0393057*inputs[235] +0.00106591*inputs[236] -0.0178783*inputs[237] +0.0111649*inputs[238] +0.0415638*inputs[239] +0.0031549*inputs[240] +0.00399799*inputs[241] +0.00520687*inputs[242] +0.0194679*inputs[243] +0.0129233*inputs[244] -0.0376939*inputs[245] +0.0156453*inputs[246] +0.00405837*inputs[247] +0.0473736*inputs[248] +0.00776436*inputs[249] -0.0100175*inputs[250] +0.0279644*inputs[251] +0.0125649*inputs[252] +0.0366802*inputs[253] +0.0135776*inputs[254] +0.00722532*inputs[255] -0.0128384*inputs[256] +0.00230387*inputs[257] -0.00837627*inputs[258] +0.00637692*inputs[259] +0.00282037*inputs[260] -0.0303699*inputs[261] -0.0156281*inputs[262] +0.00124704*inputs[263] -0.00328784*inputs[264] +0.00892766*inputs[265] -0.0261741*inputs[266] +0.05958*inputs[267] +0.0220852*inputs[268] -0.0109222*inputs[269] +0.00857502*inputs[270] +0.0537281*inputs[271] +0.00454874*inputs[272] -0.00715479*inputs[273] +0.00282037*inputs[274] +0.00282037*inputs[275] -0.0102944*inputs[276] +0.0034592*inputs[277] +0.0501737*inputs[278] -0.0013895*inputs[279] -0.0233498*inputs[280] +0.000183365*inputs[281] -0.0184314*inputs[282] -0.0305616*inputs[283] -0.0213244*inputs[284] -0.0821958*inputs[285] +0.00829769*inputs[286] +0.0257947*inputs[287] +0.016543*inputs[288] -0.0220148*inputs[289] -0.0266114*inputs[290] +0.00380592*inputs[291] +0.00842397*inputs[292] +0.0167184*inputs[293] -0.0196727*inputs[294] -0.0040418*inputs[295] +0.017796*inputs[296] -0.0015862*inputs[297] -0.0214799*inputs[298] -0.0494535*inputs[299] -0.0337381*inputs[300] -0.00950527*inputs[301] -0.00552842*inputs[302] +0.0884853*inputs[303] -0.00595885*inputs[304] +0.0431093*inputs[305] +0.0177981*inputs[306] +0.039553*inputs[307] +0.0195642*inputs[308] +0.0202722*inputs[309] +0.00230385*inputs[310] +0.00864132*inputs[311] +0.0555655*inputs[312] -0.00977814*inputs[313] -0.0411119*inputs[314] -0.0101481*inputs[315] +0.00625501*inputs[316] -0.0124227*inputs[317] -0.0013926*inputs[318] -0.0293457*inputs[319] +0.0280221*inputs[320] +0.00698982*inputs[321] +0.00529963*inputs[322] +0.0174357*inputs[323] +0.00369562*inputs[324] +0.000676736*inputs[325] +0.0060101*inputs[326] -0.00468679*inputs[327] +0.00100203*inputs[328] +0.00244123*inputs[329] +0.0127552*inputs[330] -0.01758*inputs[331] -0.029453*inputs[332] +0.00529963*inputs[333] +0.0253086*inputs[334] +0.022782*inputs[335] +0.00815152*inputs[336] -0.0160595*inputs[337] +0.0415759*inputs[338] +0.00625295*inputs[339] +0.00472708*inputs[340] -0.0497554*inputs[341] +0.000161167*inputs[342] -0.0108677*inputs[343] +0.0231499*inputs[344] -0.0305227*inputs[345] +0.0220775*inputs[346] +0.00188972*inputs[347] +0.0200913*inputs[348] -0.0146387*inputs[349] +8.49867e-05*inputs[350] -0.0404642*inputs[351] +0.00604727*inputs[352] -0.010482*inputs[353] +0.0135472*inputs[354] -0.00927326*inputs[355] -0.00456301*inputs[356] -0.0117845*inputs[357] +0.00244122*inputs[358] -0.00662955*inputs[359] -0.0139888*inputs[360] -0.000317453*inputs[361] +0.00259262*inputs[362] +0.012635*inputs[363] -0.0124147*inputs[364] +0.0174215*inputs[365] -0.00011955*inputs[366] +0.0229277*inputs[367] -0.020859*inputs[368] +0.0129247*inputs[369] +0.00244124*inputs[370] +0.00244123*inputs[371] +0.00912688*inputs[372] +0.0119789*inputs[373] -0.019924*inputs[374] +0.00978097*inputs[375] -0.0102837*inputs[376] -0.0355232*inputs[377] -0.0124383*inputs[378] -0.018138*inputs[379] -0.00302681*inputs[380] -0.0431162*inputs[381] +0.0254465*inputs[382] +0.0119506*inputs[383] +0.00131037*inputs[384] +0.00557252*inputs[385] +0.00401122*inputs[386] -0.00739618*inputs[387] +0.00244123*inputs[388] +0.0139185*inputs[389] +0.00825105*inputs[390] -0.00545354*inputs[391] +0.00244124*inputs[392] -0.0315414*inputs[393] -0.0122199*inputs[394] +0.0412737*inputs[395] -0.00434635*inputs[396] -0.0109089*inputs[397] -0.0135952*inputs[398] +0.0144012*inputs[399] -0.0297307*inputs[400] -0.0261276*inputs[401] +0.0108985*inputs[402] -0.0230735*inputs[403] +0.00654594*inputs[404] -0.000334854*inputs[405] +0.0640021*inputs[406] -0.000409944*inputs[407] +0.0279979*inputs[408] +0.012594*inputs[409] -0.000436534*inputs[410] +0.0460435*inputs[411] -0.00431737*inputs[412] +0.00244122*inputs[413] +0.0197417*inputs[414] -0.00335605*inputs[415] +0.0240102*inputs[416] -0.00087626*inputs[417] +0.0169048*inputs[418] +0.0114353*inputs[419] +0.000441212*inputs[420] +0.0306246*inputs[421] +0.00244124*inputs[422] +0.0255732*inputs[423] +0.0142467*inputs[424] +0.0184078*inputs[425] +0.0115017*inputs[426] +0.0374043*inputs[427] -0.0126882*inputs[428] -0.00892615*inputs[429] -0.0094133*inputs[430] -0.012099*inputs[431] +0.048407*inputs[432] -0.011494*inputs[433] +0.0184842*inputs[434] +0.00901236*inputs[435] +0.00534194*inputs[436] -0.00395749*inputs[437] -0.00103271*inputs[438] +0.0336153*inputs[439] +0.0037866*inputs[440] +0.00244124*inputs[441] +0.00791747*inputs[442] +0.0160922*inputs[443] +0.00716798*inputs[444] +0.0336084*inputs[445] +0.0758722*inputs[446] -0.026827*inputs[447] +0.0181388*inputs[448] +0.0118605*inputs[449] +0.00199217*inputs[450] +0.0391496*inputs[451] +0.0121062*inputs[452] +0.00928895*inputs[453] -0.0161935*inputs[454] +0.0248026*inputs[455] +0.0248026*inputs[456] -0.00501205*inputs[457] -0.00443483*inputs[458] -0.0551104*inputs[459] +0.00199217*inputs[460] +0.0305164*inputs[461] +0.0236091*inputs[462] +0.00140784*inputs[463] +0.0253789*inputs[464] +0.00199217*inputs[465] -0.0174232*inputs[466] +0.0670609*inputs[467] -0.00780602*inputs[468] -0.0079032*inputs[469] -0.0017937*inputs[470] -0.0256749*inputs[471] -0.0343326*inputs[472] +0.0134598*inputs[473] -0.00782607*inputs[474] -0.0116393*inputs[475] +0.0142657*inputs[476] +0.0101785*inputs[477] +0.0809494*inputs[478] +0.0268919*inputs[479] +0.0229511*inputs[480] -0.0240388*inputs[481] +0.00365725*inputs[482] +0.0120813*inputs[483] +0.0019922*inputs[484] +0.00308397*inputs[485] +0.00907715*inputs[486] +0.00199217*inputs[487] -0.0267172*inputs[488] +0.012359*inputs[489] +0.0939298*inputs[490] +0.00198312*inputs[491] -0.00732855*inputs[492] +0.0223128*inputs[493] -0.00277442*inputs[494] -0.00533981*inputs[495] +0.026488*inputs[496] +0.0298793*inputs[497] +0.0181233*inputs[498] -0.00421339*inputs[499] -0.0418453*inputs[500] -0.0156797*inputs[501] +0.0191539*inputs[502] +0.00199217*inputs[503] +0.0310269*inputs[504] +0.00199217*inputs[505] -0.0065819*inputs[506] -0.029898*inputs[507] -0.000815806*inputs[508] +0.0249327*inputs[509] -0.02749*inputs[510] -0.0130285*inputs[511] +0.0395922*inputs[512] +0.0157026*inputs[513] +0.00199218*inputs[514] +0.0407663*inputs[515] +0.0177427*inputs[516] +0.00199218*inputs[517] +0.00269394*inputs[518] +0.0115989*inputs[519] +0.0206192*inputs[520] +0.0101763*inputs[521] +0.00986869*inputs[522] +0.00986869*inputs[523] +0.040642*inputs[524] -0.0216838*inputs[525] +0.0131425*inputs[526] +0.011119*inputs[527] -0.00340633*inputs[528] +0.0019922*inputs[529] +0.00432175*inputs[530] -0.00863598*inputs[531] -0.00863596*inputs[532] +0.00199218*inputs[533] -0.0143502*inputs[534] -0.00911168*inputs[535] +0.0131823*inputs[536] +0.00564187*inputs[537] +0.00910414*inputs[538] -0.0147411*inputs[539] -0.0147411*inputs[540] +0.00224331*inputs[541] +0.00258322*inputs[542] -0.00165805*inputs[543] +0.0390898*inputs[544] +0.00199218*inputs[545] +0.0019922*inputs[546] -0.00544262*inputs[547] +0.0127427*inputs[548] +0.00199218*inputs[549] +0.00199221*inputs[550] +0.0330944*inputs[551] +0.0211549*inputs[552] -0.00911166*inputs[553] +0.00622831*inputs[554] +0.0019922*inputs[555] -0.00388575*inputs[556] +0.0335134*inputs[557] +0.00328024*inputs[558] +0.0459536*inputs[559] +0.00372646*inputs[560] -0.0236982*inputs[561] +0.000479015*inputs[562] +0.0400223*inputs[563] -0.040244*inputs[564] -0.0137591*inputs[565] +0.00414075*inputs[566] +0.0231703*inputs[567] +0.00199218*inputs[568] -0.00986844*inputs[569] -0.00594184*inputs[570] +0.0274761*inputs[571] +0.013154*inputs[572] -0.00447019*inputs[573] -0.0096963*inputs[574] +0.0152589*inputs[575] -0.00293734*inputs[576] +0.00198315*inputs[577] +0.00199218*inputs[578] +0.0231156*inputs[579] -0.0192675*inputs[580] +0.00140784*inputs[581] +0.00739799*inputs[582] +0.0144707*inputs[583] +0.000849261*inputs[584] +0.0279658*inputs[585] -0.00470145*inputs[586] -0.0145241*inputs[587] +0.00199221*inputs[588] +0.00199217*inputs[589] +0.0249014*inputs[590] -0.000815793*inputs[591] +0.0329116*inputs[592] +0.0052532*inputs[593] +0.0103878*inputs[594] -0.00773305*inputs[595] -0.0220399*inputs[596] -0.00159621*inputs[597] +0.00436608*inputs[598] -0.00986168*inputs[599] -0.00786634*inputs[600] +0.00160426*inputs[601] +0.00199217*inputs[602] -0.025826*inputs[603] -0.0216428*inputs[604] +0.0258374*inputs[605] +0.0299886*inputs[606] +0.00199217*inputs[607] +0.0390038*inputs[608] +0.0138537*inputs[609] -0.00805594*inputs[610] -0.0160157*inputs[611] +0.00639857*inputs[612] +0.000654366*inputs[613] +0.00199217*inputs[614] +0.00199217*inputs[615] +0.0467806*inputs[616] -0.0167935*inputs[617] +0.00837424*inputs[618] +0.00414075*inputs[619] -0.00149713*inputs[620] +0.00129043*inputs[621] +0.0113893*inputs[622] -0.00288019*inputs[623] +0.0161575*inputs[624] -0.0127079*inputs[625] +0.00199221*inputs[626] +0.0123374*inputs[627] -0.0145241*inputs[628] -0.0100842*inputs[629] +0.053517*inputs[630] -0.0288633*inputs[631] +0.00269394*inputs[632] -0.0157983*inputs[633] -0.000167813*inputs[634] -0.00153052*inputs[635] -0.000167815*inputs[636] -0.0266738*inputs[637] -0.0236114*inputs[638] -0.0130177*inputs[639] -0.0168967*inputs[640] +0.0178656*inputs[641] +0.00484375*inputs[642] +0.00199219*inputs[643] +0.00351232*inputs[644] +0.0084932*inputs[645] +0.0121062*inputs[646] -0.00149801*inputs[647] +0.00140785*inputs[648] +0.00352463*inputs[649] +0.00597766*inputs[650] +0.0117218*inputs[651] +0.0117218*inputs[652] +0.011722*inputs[653] -0.00668325*inputs[654] +0.0014079*inputs[655] +0.00140784*inputs[656] +0.00140785*inputs[657] +0.00140784*inputs[658] +0.00140783*inputs[659] +0.00140784*inputs[660] +0.00140785*inputs[661] +0.00825301*inputs[662] +0.0109293*inputs[663] +0.0109293*inputs[664] +0.00140783*inputs[665] -0.0105792*inputs[666] +0.0179277*inputs[667] +0.0179277*inputs[668] +0.0314125*inputs[669] -0.00872322*inputs[670] +0.0152562*inputs[671] +0.00140787*inputs[672] -0.0050594*inputs[673] -0.00505939*inputs[674] -0.0050594*inputs[675] +0.0114225*inputs[676] +0.0114225*inputs[677] +0.0114226*inputs[678] +0.022819*inputs[679] -0.0221016*inputs[680] -0.0221016*inputs[681] -0.0221016*inputs[682] +0.00140789*inputs[683] +0.00140788*inputs[684] +0.00140784*inputs[685] +0.00140784*inputs[686] +0.00140783*inputs[687] +0.00963718*inputs[688] +0.00963718*inputs[689] +0.00140783*inputs[690] +0.00140786*inputs[691] -0.0107116*inputs[692] -0.0107116*inputs[693] -0.0107116*inputs[694] -0.0107116*inputs[695] +0.00140783*inputs[696] +0.00140784*inputs[697] +0.00140783*inputs[698] -0.00215461*inputs[699] -0.0021546*inputs[700] +0.0125414*inputs[701] +0.00635381*inputs[702] +0.0063538*inputs[703] +0.0063538*inputs[704] +0.0063538*inputs[705] -0.00699048*inputs[706] +0.00140785*inputs[707] +0.00140783*inputs[708] +0.00140783*inputs[709] +0.00140788*inputs[710] +0.00140783*inputs[711] +0.00140783*inputs[712] +0.063528*inputs[713] +0.0238889*inputs[714] +0.0434143*inputs[715] -0.00763797*inputs[716] -0.00763792*inputs[717] +0.0014079*inputs[718] +0.00140784*inputs[719] +0.0105578*inputs[720] +0.0105577*inputs[721] -0.000730996*inputs[722] +0.00248846*inputs[723] +0.00140784*inputs[724] +0.00140788*inputs[725] +0.00140783*inputs[726] +0.00140784*inputs[727] +0.00140784*inputs[728] +0.0369733*inputs[729] -0.00293732*inputs[730] -0.00293734*inputs[731] -0.00293734*inputs[732] -0.00293733*inputs[733] -0.00293734*inputs[734] +0.00140783*inputs[735] +0.00140783*inputs[736] +0.00140785*inputs[737] -0.0177176*inputs[738] +0.00140783*inputs[739] +0.00140783*inputs[740] +0.00140783*inputs[741] -0.0154957*inputs[742] -0.0154958*inputs[743] -0.00596123*inputs[744] -0.0301458*inputs[745] -0.0278147*inputs[746] -0.0278147*inputs[747] +0.0133708*inputs[748] +0.0133708*inputs[749] -0.00679467*inputs[750] -0.00679466*inputs[751] -0.0127953*inputs[752] +0.00140783*inputs[753] +0.00140783*inputs[754] -0.00460313*inputs[755] -0.00460313*inputs[756] -0.00460311*inputs[757] -0.00460313*inputs[758] -0.00994268*inputs[759] -0.00994274*inputs[760] +0.00140783*inputs[761] +0.00166991*inputs[762] +0.00166991*inputs[763] +0.00166992*inputs[764] +0.00166993*inputs[765] +0.00166993*inputs[766] +0.00166991*inputs[767] +0.0107967*inputs[768] +0.0107967*inputs[769] +0.00140784*inputs[770] -0.0121765*inputs[771] -0.0121765*inputs[772] +0.00140785*inputs[773] +0.00140786*inputs[774] +0.00140783*inputs[775] +0.00140783*inputs[776] +0.00140785*inputs[777] +0.00140785*inputs[778] +0.00140783*inputs[779] +0.00140789*inputs[780] +0.0531955*inputs[781] +0.0237254*inputs[782] +0.0237254*inputs[783] +0.0082169*inputs[784] +0.0082169*inputs[785] +0.0206432*inputs[786] +0.00140783*inputs[787] +0.00140785*inputs[788] +0.00140783*inputs[789] -0.0147168*inputs[790] +0.00140784*inputs[791] +0.000295623*inputs[792] +0.00140784*inputs[793] +0.00140783*inputs[794] -0.00913501*inputs[795] -0.00913501*inputs[796] -0.0236309*inputs[797] +0.00140783*inputs[798] +0.00140783*inputs[799] +0.0014079*inputs[800] -0.00182867*inputs[801] -0.00182866*inputs[802] +0.0124148*inputs[803] +0.00140784*inputs[804] +0.00641891*inputs[805] +0.00140785*inputs[806] -0.00612943*inputs[807] +0.00140789*inputs[808] +0.00140786*inputs[809] +0.00140783*inputs[810] +0.00140783*inputs[811] +0.00641891*inputs[812] -0.0392699*inputs[813] -0.00285494*inputs[814] -0.00285492*inputs[815] +0.00140785*inputs[816] -0.00612949*inputs[817] +0.00140783*inputs[818] +0.00140784*inputs[819] +0.00140783*inputs[820] +0.00140783*inputs[821] +0.0314117*inputs[822] +0.0267368*inputs[823] +0.00453569*inputs[824] +0.00140784*inputs[825] +0.00140783*inputs[826] -0.00612947*inputs[827] +0.00453573*inputs[828] -0.0328657*inputs[829] -0.0142876*inputs[830] -0.00736377*inputs[831] -0.00736377*inputs[832] -0.012673*inputs[833] +0.00453573*inputs[834] -0.00736377*inputs[835] -0.00627244*inputs[836] -0.00690059*inputs[837] +0.00140783*inputs[838] -0.0143068*inputs[839] +0.00140783*inputs[840] +0.000816244*inputs[841] -0.0185206*inputs[842] +0.00978766*inputs[843] +0.00978766*inputs[844] +0.00978764*inputs[845] +0.00978764*inputs[846] +0.00978764*inputs[847] -0.0247708*inputs[848] +0.0148441*inputs[849] +0.014844*inputs[850] -0.0247708*inputs[851] +0.0234858*inputs[852] +0.0129761*inputs[853] -0.0139485*inputs[854] -0.00893405*inputs[855] +0.00140789*inputs[856] +0.00140783*inputs[857] +0.00140783*inputs[858] -0.00205275*inputs[859] +0.0014079*inputs[860] +0.00140783*inputs[861] +0.00140786*inputs[862] +0.00140787*inputs[863] +0.00140785*inputs[864] -0.01785*inputs[865] +0.00322859*inputs[866] +0.0129763*inputs[867] +0.0155361*inputs[868] +0.00140785*inputs[869] +0.00801313*inputs[870] +0.00801313*inputs[871] +0.00140784*inputs[872] +0.00732554*inputs[873] +0.00732554*inputs[874] +0.00732554*inputs[875] +0.00140789*inputs[876] -0.00627243*inputs[877] +0.0227591*inputs[878] +0.0237934*inputs[879] +0.0227591*inputs[880] -0.0105407*inputs[881] +0.0366414*inputs[882] +0.00140784*inputs[883] +0.00140783*inputs[884] -0.00339024*inputs[885] -0.0033902*inputs[886] +0.00140783*inputs[887] +0.00140785*inputs[888] +0.00140784*inputs[889] +0.00140786*inputs[890] +0.00140791*inputs[891] +0.00140783*inputs[892] +0.00140784*inputs[893] +0.00140784*inputs[894] +0.00140787*inputs[895] +0.00140784*inputs[896] +0.00666514*inputs[897] +0.00140783*inputs[898] +0.00133028*inputs[899] +0.00666512*inputs[900] -0.00627243*inputs[901] -0.0082756*inputs[902] +0.019023*inputs[903] +0.00140784*inputs[904] -0.00827567*inputs[905] -0.00827567*inputs[906] +0.000881385*inputs[907] +0.000881361*inputs[908] +0.00140783*inputs[909] +0.00140783*inputs[910] +0.00140783*inputs[911] -0.0295287*inputs[912] +0.0214307*inputs[913] +0.00140784*inputs[914] +0.00140784*inputs[915] +0.0190231*inputs[916] +0.00123879*inputs[917] +0.00140783*inputs[918] +0.00123882*inputs[919] +0.00123885*inputs[920] +0.0190231*inputs[921] +0.0381218*inputs[922] -0.014329*inputs[923] +0.0123164*inputs[924] +0.00140783*inputs[925] +0.00140784*inputs[926] +0.00140784*inputs[927] +0.00140783*inputs[928] +0.0322959*inputs[929] +0.00140783*inputs[930] +0.00140787*inputs[931] +0.0313756*inputs[932] +0.0120813*inputs[933] -0.00755157*inputs[934] +0.00140785*inputs[935] +0.0120813*inputs[936] +0.00805707*inputs[937] -0.00755161*inputs[938] +0.0157242*inputs[939] +0.00376149*inputs[940] +0.0037615*inputs[941] +0.0037615*inputs[942] +0.0037615*inputs[943] -0.000628687*inputs[944] +0.0179279*inputs[945] -0.000628671*inputs[946] -0.0242976*inputs[947] +0.00140784*inputs[948] -0.00805358*inputs[949] -0.00805363*inputs[950] +0.0179279*inputs[951] +0.0220232*inputs[952] +0.00140784*inputs[953] +0.00140784*inputs[954] -0.0102865*inputs[955] -0.0102866*inputs[956] -0.0102866*inputs[957] -0.0102866*inputs[958] +0.0203968*inputs[959] +0.00140784*inputs[960] +0.00140784*inputs[961] -0.010581*inputs[962] -0.010581*inputs[963] -0.00405768*inputs[964] -0.0299745*inputs[965] +0.00140788*inputs[966] +0.0014079*inputs[967] +0.00140784*inputs[968] +0.0314116*inputs[969] +0.00140784*inputs[970] -0.0188619*inputs[971] +0.00140784*inputs[972] +0.00140783*inputs[973] +0.00140784*inputs[974] +0.00140785*inputs[975] +0.0520548*inputs[976] +0.0191671*inputs[977] -0.0142316*inputs[978] +0.00140783*inputs[979] +0.0218351*inputs[980] +0.00140783*inputs[981] -0.0244901*inputs[982] -0.0244902*inputs[983] +0.00140785*inputs[984] +0.00140786*inputs[985] +0.00140783*inputs[986] -0.000507929*inputs[987] +0.00140783*inputs[988] +0.00140784*inputs[989] +0.00140785*inputs[990] +0.0080034*inputs[991] +0.0158984*inputs[992] +0.00140784*inputs[993] -0.00308813*inputs[994] +0.00140786*inputs[995] +0.0199653*inputs[996] -0.0311134*inputs[997] +0.00800341*inputs[998] +0.00140783*inputs[999] +0.00140785*inputs[1000] +0.00140784*inputs[1001] +0.0107349*inputs[1002] +0.00476345*inputs[1003] +0.00800342*inputs[1004] +0.00140783*inputs[1005] +0.00140789*inputs[1006] +0.00140785*inputs[1007] +0.0228408*inputs[1008] -0.0123388*inputs[1009] -0.0123388*inputs[1010] -0.0123388*inputs[1011] +0.00140783*inputs[1012] +0.00140784*inputs[1013] +0.00140783*inputs[1014] +0.00140783*inputs[1015] +0.00933026*inputs[1016] +0.00933026*inputs[1017] -0.0153476*inputs[1018] -0.0153475*inputs[1019] +0.0242093*inputs[1020] +0.000859608*inputs[1021] +0.00140783*inputs[1022] +0.00140783*inputs[1023] +0.00800341*inputs[1024] +0.00357985*inputs[1025] +0.00357985*inputs[1026] +0.00357988*inputs[1027] +0.00140784*inputs[1028] +0.00140785*inputs[1029] +0.00140784*inputs[1030] +0.0080034*inputs[1031] -0.00622298*inputs[1032] -0.0144902*inputs[1033] -0.00622293*inputs[1034] +0.016123*inputs[1035] +0.0161229*inputs[1036] +0.016123*inputs[1037] +0.0161229*inputs[1038] +0.0161229*inputs[1039] +0.00140791*inputs[1040] +0.00140785*inputs[1041] +0.00140783*inputs[1042] -0.0114487*inputs[1043] +0.0167093*inputs[1044] +0.0167093*inputs[1045] +0.0167093*inputs[1046] -0.0114487*inputs[1047] -0.0108296*inputs[1048] +0.00140783*inputs[1049] +0.017383*inputs[1050] -0.0108296*inputs[1051] -0.0114486*inputs[1052] +0.00140785*inputs[1053] -0.000483179*inputs[1054] -0.000483179*inputs[1055] -0.000483175*inputs[1056] -0.000483138*inputs[1057] +0.014987*inputs[1058] -0.0327301*inputs[1059] +0.00140784*inputs[1060] +0.00140783*inputs[1061] +0.00140783*inputs[1062] -0.0153571*inputs[1063] -0.0347398*inputs[1064] +0.00140783*inputs[1065] +0.0114606*inputs[1066] +0.0114606*inputs[1067] +0.00140788*inputs[1068] +0.00140784*inputs[1069] +0.00140786*inputs[1070] +0.00140785*inputs[1071] +0.00140783*inputs[1072] +0.00140784*inputs[1073] +0.00140784*inputs[1074] +0.00140783*inputs[1075] +0.00140784*inputs[1076] +0.00140787*inputs[1077] -0.00507405*inputs[1078] -0.0251458*inputs[1079] -0.0185125*inputs[1080] -0.00507405*inputs[1081] -0.00507406*inputs[1082] +0.00444492*inputs[1083] +0.00444492*inputs[1084] +0.00444495*inputs[1085] +0.0014079*inputs[1086] +0.00140783*inputs[1087] -0.00507406*inputs[1088] +0.0388772*inputs[1089] +0.00140787*inputs[1090] -0.0108296*inputs[1091] +0.0201731*inputs[1092] +0.00140786*inputs[1093] -0.0104141*inputs[1094] +0.00140789*inputs[1095] -0.0108296*inputs[1096] +0.00140783*inputs[1097] +0.00140785*inputs[1098] +0.000397549*inputs[1099] +0.000397552*inputs[1100] +0.0014079*inputs[1101] +0.00140785*inputs[1102] -0.0107413*inputs[1103] -0.0194515*inputs[1104] +0.00140784*inputs[1105] +0.00140783*inputs[1106] +0.00140784*inputs[1107] +0.00140783*inputs[1108] +0.0141081*inputs[1109] -0.00375175*inputs[1110] -0.0305586*inputs[1111] +0.0160309*inputs[1112] +0.00140783*inputs[1113] +0.0148054*inputs[1114] +0.00140783*inputs[1115] +0.00627298*inputs[1116] +0.0288769*inputs[1117] -0.0099186*inputs[1118] +0.00140783*inputs[1119] +0.00140785*inputs[1120] +0.00140784*inputs[1121] +0.00140789*inputs[1122] +0.00140783*inputs[1123] +0.00140784*inputs[1124] +0.00140788*inputs[1125] +0.00552575*inputs[1126] +0.00552574*inputs[1127] +0.00140783*inputs[1128] +0.00140786*inputs[1129] +0.0240461*inputs[1130] +0.0374295*inputs[1131] -0.0258846*inputs[1132] +0.00834588*inputs[1133] +0.0083459*inputs[1134] +0.0106964*inputs[1135] +0.0106964*inputs[1136] +0.0106964*inputs[1137] +0.0160743*inputs[1138] +0.00140784*inputs[1139] +0.00140783*inputs[1140] -0.00532976*inputs[1141] +0.0148055*inputs[1142] +0.00239984*inputs[1143] +0.00239978*inputs[1144] +0.00239978*inputs[1145] +0.00239979*inputs[1146] -0.0205271*inputs[1147] -0.0205271*inputs[1148] -0.00857834*inputs[1149] +0.00140784*inputs[1150] +0.00140787*inputs[1151] +0.00140789*inputs[1152] -0.00698963*inputs[1153] -0.00698963*inputs[1154] -0.00698962*inputs[1155] +0.0155924*inputs[1156] +0.023201*inputs[1157] +0.0401583*inputs[1158] +0.00140784*inputs[1159] +0.00140784*inputs[1160] -0.00108244*inputs[1161] +0.00140783*inputs[1162] -0.0233213*inputs[1163] +0.00140783*inputs[1164] +0.00140783*inputs[1165] +0.00140784*inputs[1166] +0.0237124*inputs[1167] +0.00140784*inputs[1168] +0.00140783*inputs[1169] +0.00140785*inputs[1170] +0.00140784*inputs[1171] +0.00140784*inputs[1172] -0.00164528*inputs[1173] -0.00820189*inputs[1174] -0.00820179*inputs[1175] -0.00820192*inputs[1176] -0.00820187*inputs[1177] -0.012365*inputs[1178] -0.012365*inputs[1179] -0.0123649*inputs[1180] -0.0123649*inputs[1181] +0.00140783*inputs[1182] +0.00140786*inputs[1183] +0.00140786*inputs[1184] +0.00140785*inputs[1185] +0.0336998*inputs[1186] +0.00140788*inputs[1187] +0.0014079*inputs[1188] +0.00140783*inputs[1189] -0.0297714*inputs[1190] +0.00140783*inputs[1191] -0.00352431*inputs[1192] +0.0326325*inputs[1193] -0.0268312*inputs[1194] +0.00140785*inputs[1195] +0.00140789*inputs[1196] -0.0303128*inputs[1197] -0.0198085*inputs[1198] -0.0198086*inputs[1199] +0.0238854*inputs[1200] +0.00140783*inputs[1201] +0.00140789*inputs[1202] +0.00140783*inputs[1203] +0.00140783*inputs[1204] +0.0221571*inputs[1205] -0.00764231*inputs[1206] -0.00764229*inputs[1207] +0.00140784*inputs[1208] +0.00140785*inputs[1209] +0.00140784*inputs[1210] +0.00140786*inputs[1211] +0.00140783*inputs[1212] +0.00140785*inputs[1213] +0.00140783*inputs[1214] +0.00140785*inputs[1215] +0.00140783*inputs[1216] +0.00140786*inputs[1217] +0.00140784*inputs[1218] -0.0209054*inputs[1219] -0.0209054*inputs[1220] -0.0209054*inputs[1221] +0.0128999*inputs[1222] +0.0128999*inputs[1223] +0.0128999*inputs[1224] +0.0196422*inputs[1225] +0.0196422*inputs[1226] +0.00140786*inputs[1227] +0.00140784*inputs[1228] +0.00140783*inputs[1229] +0.00543856*inputs[1230] +0.00543856*inputs[1231] +0.0314114*inputs[1232] +0.00140783*inputs[1233] +0.00140784*inputs[1234] +0.00140786*inputs[1235] -0.0316389*inputs[1236] +0.00140784*inputs[1237] +0.0166039*inputs[1238] +0.0168582*inputs[1239] +0.0168583*inputs[1240] +0.00140785*inputs[1241] +0.00140783*inputs[1242] +0.00140784*inputs[1243] +0.00140784*inputs[1244] +0.00140783*inputs[1245] +0.00712338*inputs[1246] +0.00712344*inputs[1247] +0.00712338*inputs[1248] -0.0142952*inputs[1249] +0.00140785*inputs[1250] +0.00140785*inputs[1251] +0.0075832*inputs[1252] +0.00140785*inputs[1253] +0.0181742*inputs[1254] +0.00601735*inputs[1255] +0.00601737*inputs[1256] +0.0163339*inputs[1257] +0.0105971*inputs[1258] +0.0105971*inputs[1259] -0.0035255*inputs[1260] -0.00450504*inputs[1261] -0.00352545*inputs[1262] -0.0035255*inputs[1263] -0.0035255*inputs[1264] +0.0329051*inputs[1265] -0.0367984*inputs[1266] +0.00140783*inputs[1267] +0.00140784*inputs[1268] +0.00140784*inputs[1269] -0.0222557*inputs[1270] -0.0192023*inputs[1271] +0.00739199*inputs[1272] +0.007392*inputs[1273] -0.00975718*inputs[1274] -0.00975718*inputs[1275] +0.00140783*inputs[1276] +0.00140785*inputs[1277] -0.0224634*inputs[1278] +0.00140789*inputs[1279] +0.00140784*inputs[1280] -0.00450504*inputs[1281] +0.00140789*inputs[1282] +0.00140785*inputs[1283] +0.0344651*inputs[1284] +0.0127238*inputs[1285] +0.046441*inputs[1286] +0.00140788*inputs[1287] +0.00140784*inputs[1288] +0.00140784*inputs[1289] +0.0085104*inputs[1290] +0.0120092*inputs[1291] +0.0157041*inputs[1292] +0.0320031*inputs[1293] +0.0119501*inputs[1294] +0.0061106*inputs[1295] +0.00140784*inputs[1296] +0.0120093*inputs[1297] -0.0316973*inputs[1298] +0.00140785*inputs[1299] +0.00140784*inputs[1300] +0.00140786*inputs[1301] +0.00140783*inputs[1302] -0.00443486*inputs[1303] +0.00140783*inputs[1304] -0.00984334*inputs[1305] -0.00984336*inputs[1306] -0.00984336*inputs[1307] +0.0014079*inputs[1308] -0.00895597*inputs[1309] +0.00140783*inputs[1310] +0.00140784*inputs[1311] +0.00140786*inputs[1312] +0.00408101*inputs[1313] +0.004081*inputs[1314] +0.0014079*inputs[1315] +0.0014079*inputs[1316] +0.00508272*inputs[1317] +0.00140784*inputs[1318] +0.0164213*inputs[1319] -0.0309454*inputs[1320] +0.00140783*inputs[1321] +0.00634263*inputs[1322] +0.00140783*inputs[1323] +0.00140786*inputs[1324] -0.0391731*inputs[1325] -0.00910122*inputs[1326] +0.00140785*inputs[1327] +0.00140784*inputs[1328] +0.00140789*inputs[1329] +0.00140783*inputs[1330] +0.0153569*inputs[1331] +0.00140783*inputs[1332] +0.00140785*inputs[1333] +0.00140785*inputs[1334] +0.00140783*inputs[1335] -0.0134649*inputs[1336] -0.0134649*inputs[1337] -0.0134649*inputs[1338] +0.010899*inputs[1339] +0.010899*inputs[1340] +0.00140784*inputs[1341] +0.00753776*inputs[1342] +0.00140784*inputs[1343] +0.00140783*inputs[1344] +0.00140786*inputs[1345] +0.0171852*inputs[1346] +0.00140786*inputs[1347] +0.00140783*inputs[1348] +0.00140783*inputs[1349] +0.00140783*inputs[1350] +0.00140783*inputs[1351] +0.0014079*inputs[1352] -0.012442*inputs[1353] -0.012442*inputs[1354] +0.00140784*inputs[1355] +0.00140786*inputs[1356] -0.0347397*inputs[1357] -0.0376998*inputs[1358] +0.00140783*inputs[1359] +0.00140783*inputs[1360] +0.00140783*inputs[1361] +0.00140783*inputs[1362] +0.00140785*inputs[1363] -0.023046*inputs[1364] -0.023046*inputs[1365] -0.000458324*inputs[1366] -0.000458405*inputs[1367] -0.000458402*inputs[1368] -0.000458372*inputs[1369] -0.000458404*inputs[1370] +0.0231522*inputs[1371] +0.00682594*inputs[1372] +0.00682595*inputs[1373] -0.00836347*inputs[1374] +0.00669931*inputs[1375] +0.00669933*inputs[1376] +0.0125392*inputs[1377] +0.0497102*inputs[1378] -0.00194072*inputs[1379] +0.00140784*inputs[1380] +0.00140787*inputs[1381] +0.00140784*inputs[1382] +0.00140784*inputs[1383] +0.0346058*inputs[1384] +0.0165566*inputs[1385] +0.0165565*inputs[1386] +0.00140787*inputs[1387] +0.00140784*inputs[1388] +0.00140784*inputs[1389] +0.0299602*inputs[1390] +0.00140784*inputs[1391] +0.00140789*inputs[1392] +0.00140784*inputs[1393] +0.00140785*inputs[1394] +0.00140784*inputs[1395] 
		combinations[2] = 0.0594318 +0.0644211*inputs[0] +0.0907325*inputs[1] +0.211331*inputs[2] +0.14979*inputs[3] +0.146827*inputs[4] +0.0753369*inputs[5] +0.0122288*inputs[6] +0.0277868*inputs[7] +0.0490871*inputs[8] +0.0018289*inputs[9] +0.00161281*inputs[10] +0.072894*inputs[11] +0.0349555*inputs[12] +0.0952137*inputs[13] +0.178624*inputs[14] +0.0378134*inputs[15] +0.00127731*inputs[16] +0.0185462*inputs[17] +0.106753*inputs[18] -0.107746*inputs[19] -0.000980149*inputs[20] +0.0144652*inputs[21] +0.029504*inputs[22] +0.000506862*inputs[23] -0.00484197*inputs[24] -0.0244282*inputs[25] +0.124227*inputs[26] -0.0231257*inputs[27] +0.00398853*inputs[28] +0.00627161*inputs[29] +0.0154054*inputs[30] -0.0332086*inputs[31] -0.0965173*inputs[32] +0.0734522*inputs[33] -0.0817903*inputs[34] +0.102083*inputs[35] +0.0858392*inputs[36] +0.00811849*inputs[37] -0.010678*inputs[38] -0.130139*inputs[39] +0.0312871*inputs[40] +0.283341*inputs[41] +0.00980835*inputs[42] -0.0408132*inputs[43] +0.0348491*inputs[44] +0.00197899*inputs[45] -0.0258679*inputs[46] -0.0679441*inputs[47] -0.0798301*inputs[48] +0.0899808*inputs[49] +0.003214*inputs[50] +0.00659327*inputs[51] +0.00599109*inputs[52] +0.0155315*inputs[53] -0.00840914*inputs[54] -0.0600934*inputs[55] +0.0121916*inputs[56] +0.0348065*inputs[57] -0.118825*inputs[58] -0.00163959*inputs[59] -0.138304*inputs[60] -0.0290917*inputs[61] -0.0299026*inputs[62] -0.037413*inputs[63] +0.000664578*inputs[64] -0.1553*inputs[65] -0.0842787*inputs[66] +0.0476115*inputs[67] -0.0420718*inputs[68] +0.000153899*inputs[69] +0.00996854*inputs[70] -0.00322495*inputs[71] +0.011636*inputs[72] -0.0107973*inputs[73] -0.0893441*inputs[74] +0.0116624*inputs[75] +0.0989711*inputs[76] +0.0168345*inputs[77] +0.0203427*inputs[78] +0.0119968*inputs[79] -0.0330449*inputs[80] -0.040432*inputs[81] +0.0126811*inputs[82] +0.0223528*inputs[83] +0.01967*inputs[84] -0.0443637*inputs[85] +0.0433499*inputs[86] -0.0166586*inputs[87] -0.0334926*inputs[88] -0.0146402*inputs[89] -0.007542*inputs[90] -0.0243986*inputs[91] -0.0337426*inputs[92] -0.0221168*inputs[93] -0.103821*inputs[94] -0.0877434*inputs[95] -0.00218071*inputs[96] +0.0322806*inputs[97] -0.0154464*inputs[98] +0.0377143*inputs[99] +0.00638922*inputs[100] -0.0327497*inputs[101] -0.0218962*inputs[102] -0.0141791*inputs[103] -0.0483431*inputs[104] +0.00303845*inputs[105] -0.0804943*inputs[106] +0.00265497*inputs[107] -0.0845081*inputs[108] +0.00830922*inputs[109] +0.0441553*inputs[110] +0.0581242*inputs[111] -0.0205366*inputs[112] +0.0199393*inputs[113] -0.0630644*inputs[114] +0.0187988*inputs[115] -0.00418607*inputs[116] -0.0171197*inputs[117] -0.033653*inputs[118] -0.0178474*inputs[119] -0.0320954*inputs[120] +0.0781095*inputs[121] +0.069827*inputs[122] -0.0843999*inputs[123] -0.0289792*inputs[124] -0.0134149*inputs[125] -0.00371919*inputs[126] -0.00353511*inputs[127] +0.00688093*inputs[128] +0.00417668*inputs[129] +0.00832178*inputs[130] +0.117435*inputs[131] +0.0207419*inputs[132] +0.0113256*inputs[133] -0.0215784*inputs[134] -0.0257135*inputs[135] -0.0790155*inputs[136] -0.0056508*inputs[137] -0.0223883*inputs[138] -0.085866*inputs[139] +0.0128119*inputs[140] +0.0222152*inputs[141] +0.0283828*inputs[142] +0.0282607*inputs[143] -0.0151675*inputs[144] +0.0101821*inputs[145] +0.00302047*inputs[146] -0.018742*inputs[147] -0.025967*inputs[148] -0.0389773*inputs[149] -0.0193037*inputs[150] -0.0150987*inputs[151] -0.0364608*inputs[152] -0.00142956*inputs[153] -0.0650909*inputs[154] -0.0175755*inputs[155] +0.00158933*inputs[156] -0.00966735*inputs[157] -0.0256648*inputs[158] +0.0626587*inputs[159] -0.00464662*inputs[160] -0.0141351*inputs[161] -5.13742e-05*inputs[162] -0.0241938*inputs[163] -0.00802438*inputs[164] -0.0175404*inputs[165] +0.0198312*inputs[166] +0.00639632*inputs[167] -0.00094328*inputs[168] -0.0220558*inputs[169] -0.0111548*inputs[170] -0.0384132*inputs[171] +0.00497812*inputs[172] +0.0212203*inputs[173] -0.0165029*inputs[174] +0.0515824*inputs[175] +0.0112766*inputs[176] +0.0024543*inputs[177] -0.071328*inputs[178] +0.0434721*inputs[179] -0.0567817*inputs[180] +0.00238731*inputs[181] +0.00131044*inputs[182] +0.0121378*inputs[183] -0.0128613*inputs[184] -0.00271579*inputs[185] -0.0122357*inputs[186] -0.012178*inputs[187] +0.015185*inputs[188] -0.00905141*inputs[189] -0.0252568*inputs[190] +0.00469718*inputs[191] +0.0344201*inputs[192] -0.00697274*inputs[193] +0.0503626*inputs[194] +0.0190208*inputs[195] +0.0196805*inputs[196] -0.0369817*inputs[197] +0.0340413*inputs[198] -0.0213226*inputs[199] -0.000392768*inputs[200] +0.00407199*inputs[201] -0.00434492*inputs[202] -0.0143286*inputs[203] +0.0261025*inputs[204] -0.0422569*inputs[205] +0.0224797*inputs[206] -0.0266541*inputs[207] -0.0180742*inputs[208] -0.0532571*inputs[209] +0.000430698*inputs[210] -0.00580621*inputs[211] -0.0276158*inputs[212] -0.00351181*inputs[213] +0.0172494*inputs[214] +0.0544752*inputs[215] +0.0265281*inputs[216] -0.0221625*inputs[217] +0.00976992*inputs[218] -0.00718851*inputs[219] +0.0169242*inputs[220] -0.0181031*inputs[221] +0.0150141*inputs[222] +0.00292933*inputs[223] -0.0434127*inputs[224] -0.0556261*inputs[225] -0.000627765*inputs[226] -0.00822131*inputs[227] +0.00973161*inputs[228] +0.0168329*inputs[229] +0.01909*inputs[230] +0.00402565*inputs[231] -0.00444385*inputs[232] +0.00920955*inputs[233] +0.000943867*inputs[234] +0.039096*inputs[235] -0.00161014*inputs[236] +0.0177818*inputs[237] -0.0116228*inputs[238] -0.0419977*inputs[239] -0.00421026*inputs[240] -0.00236448*inputs[241] -0.00625254*inputs[242] -0.0183939*inputs[243] -0.0135976*inputs[244] +0.0348562*inputs[245] -0.0183275*inputs[246] -0.00451816*inputs[247] -0.0482762*inputs[248] -0.00809147*inputs[249] +0.0076597*inputs[250] -0.0290276*inputs[251] -0.0136574*inputs[252] -0.0369451*inputs[253] -0.0158829*inputs[254] -0.00840191*inputs[255] +0.0122201*inputs[256] -0.00300715*inputs[257] +0.00939273*inputs[258] -0.00555263*inputs[259] -0.0037638*inputs[260] +0.0296057*inputs[261] +0.00296291*inputs[262] -0.00447148*inputs[263] +0.00284844*inputs[264] -0.0108093*inputs[265] +0.0251967*inputs[266] -0.0599025*inputs[267] -0.0230534*inputs[268] +0.010824*inputs[269] -0.0090864*inputs[270] -0.0548759*inputs[271] -0.00476119*inputs[272] +0.00621268*inputs[273] -0.0037638*inputs[274] -0.00376381*inputs[275] +0.0102925*inputs[276] -0.00827227*inputs[277] -0.0519804*inputs[278] +0.000125292*inputs[279] +0.0218856*inputs[280] -0.025167*inputs[281] +0.0190164*inputs[282] +0.0291417*inputs[283] +0.0208122*inputs[284] +0.0765613*inputs[285] -0.0510639*inputs[286] -0.025818*inputs[287] -0.0170851*inputs[288] +0.0203181*inputs[289] +0.0223102*inputs[290] -0.00296826*inputs[291] -0.00904619*inputs[292] -0.0168904*inputs[293] +0.0181692*inputs[294] +0.00301263*inputs[295] -0.0181744*inputs[296] +0.00151994*inputs[297] +0.0200087*inputs[298] +0.0470136*inputs[299] +0.0348517*inputs[300] +0.0103307*inputs[301] +0.00478024*inputs[302] -0.0869839*inputs[303] +0.00635756*inputs[304] -0.0424808*inputs[305] -0.0189991*inputs[306] -0.0397881*inputs[307] -0.0192865*inputs[308] -0.0211422*inputs[309] -0.00300715*inputs[310] -0.00872194*inputs[311] -0.0619082*inputs[312] +0.00889883*inputs[313] +0.0401507*inputs[314] +0.0104195*inputs[315] -0.00662665*inputs[316] +0.00191603*inputs[317] +0.00422184*inputs[318] +0.0302079*inputs[319] -0.0278712*inputs[320] -0.00543788*inputs[321] -0.00483122*inputs[322] -0.0170152*inputs[323] -0.00273238*inputs[324] -0.000111803*inputs[325] -0.00645131*inputs[326] +0.00368542*inputs[327] +0.000921631*inputs[328] -0.00325786*inputs[329] -0.0134523*inputs[330] +0.0178855*inputs[331] +0.0289061*inputs[332] -0.00483121*inputs[333] -0.0262579*inputs[334] -0.024041*inputs[335] -0.00825162*inputs[336] +0.0163547*inputs[337] -0.0421826*inputs[338] -0.00671193*inputs[339] -0.00765384*inputs[340] +0.0479292*inputs[341] -0.000540805*inputs[342] +0.0106113*inputs[343] -0.0242294*inputs[344] +0.0318194*inputs[345] -0.0128277*inputs[346] -0.00252189*inputs[347] -0.0213891*inputs[348] +0.015816*inputs[349] -0.0009783*inputs[350] +0.0386498*inputs[351] -0.00648045*inputs[352] +0.00951167*inputs[353] -0.0135682*inputs[354] +0.00857867*inputs[355] -0.00240319*inputs[356] +0.0114697*inputs[357] -0.00325785*inputs[358] +0.00534648*inputs[359] +0.0144098*inputs[360] -0.000314907*inputs[361] -0.0068381*inputs[362] -0.0134378*inputs[363] +0.0116341*inputs[364] -0.0172857*inputs[365] -0.00122148*inputs[366] -0.0236888*inputs[367] +0.0195196*inputs[368] -0.0151059*inputs[369] -0.00325784*inputs[370] -0.00325787*inputs[371] -0.00978421*inputs[372] -0.0121882*inputs[373] +0.0177316*inputs[374] -0.0107266*inputs[375] +0.00948096*inputs[376] +0.0353799*inputs[377] +0.0114144*inputs[378] +0.0178178*inputs[379] +0.00204339*inputs[380] +0.0415875*inputs[381] -0.0261351*inputs[382] -0.0131585*inputs[383] -0.00892242*inputs[384] -0.00593276*inputs[385] -0.00471615*inputs[386] +0.00650206*inputs[387] -0.00325789*inputs[388] -0.0154411*inputs[389] -0.00854932*inputs[390] +0.00453495*inputs[391] -0.00325784*inputs[392] +0.0307337*inputs[393] +0.0178569*inputs[394] -0.0417968*inputs[395] +0.00365287*inputs[396] +0.010003*inputs[397] +0.0171365*inputs[398] -0.0122297*inputs[399] -0.0019162*inputs[400] +0.0289425*inputs[401] -0.0115752*inputs[402] +0.0249145*inputs[403] -0.00532235*inputs[404] +0.0002908*inputs[405] -0.0644637*inputs[406] -0.00040688*inputs[407] -0.0291982*inputs[408] -0.0131019*inputs[409] +0.0031746*inputs[410] -0.0462861*inputs[411] +0.00410717*inputs[412] -0.00325784*inputs[413] -0.0206089*inputs[414] +0.00284479*inputs[415] -0.0245111*inputs[416] +0.000406095*inputs[417] -0.017808*inputs[418] -0.0108075*inputs[419] -0.000754103*inputs[420] -0.0309731*inputs[421] -0.00325785*inputs[422] -0.0273875*inputs[423] -0.0140073*inputs[424] -0.0190634*inputs[425] -0.0120932*inputs[426] -0.0368243*inputs[427] +0.0114883*inputs[428] +0.00851619*inputs[429] +0.0091293*inputs[430] +0.0113564*inputs[431] -0.0471702*inputs[432] +0.010872*inputs[433] -0.0194317*inputs[434] -0.00979373*inputs[435] -0.00517122*inputs[436] +0.00402842*inputs[437] +0.000612286*inputs[438] -0.0344051*inputs[439] -0.00365567*inputs[440] -0.00325784*inputs[441] -0.00996003*inputs[442] -0.0159896*inputs[443] -0.00682073*inputs[444] -0.0319801*inputs[445] -0.0760118*inputs[446] +0.0249145*inputs[447] -0.0181002*inputs[448] -0.0113226*inputs[449] -0.00265861*inputs[450] -0.0396693*inputs[451] -0.0128258*inputs[452] -0.0096338*inputs[453] +0.015571*inputs[454] -0.0250136*inputs[455] -0.0250135*inputs[456] +0.00471222*inputs[457] +0.00481391*inputs[458] +0.0547133*inputs[459] -0.00265861*inputs[460] -0.0297099*inputs[461] -0.022731*inputs[462] -0.00187884*inputs[463] -0.0222372*inputs[464] -0.00265861*inputs[465] +0.0179508*inputs[466] -0.0663245*inputs[467] +0.00717021*inputs[468] +0.00880291*inputs[469] +0.00119549*inputs[470] +0.000442479*inputs[471] +0.0376387*inputs[472] -0.0131121*inputs[473] +0.0112774*inputs[474] +0.00856286*inputs[475] -0.0146067*inputs[476] -0.00928752*inputs[477] -0.0922011*inputs[478] -0.0280999*inputs[479] -0.0222448*inputs[480] +0.0228113*inputs[481] -0.00424412*inputs[482] -0.0123792*inputs[483] -0.00265863*inputs[484] -0.00364399*inputs[485] -0.00970763*inputs[486] -0.00265866*inputs[487] +0.0260248*inputs[488] -0.01326*inputs[489] -0.0807684*inputs[490] -0.002646*inputs[491] +0.00627009*inputs[492] -0.0224415*inputs[493] +0.00261211*inputs[494] +0.00474953*inputs[495] -0.0260963*inputs[496] -0.0308738*inputs[497] -0.0190846*inputs[498] +0.00338802*inputs[499] +0.0405358*inputs[500] +0.0199115*inputs[501] -0.0186258*inputs[502] -0.00265861*inputs[503] -0.0297847*inputs[504] -0.00265864*inputs[505] -0.00161241*inputs[506] +0.0300531*inputs[507] +0.00193771*inputs[508] -0.0245108*inputs[509] +0.0217365*inputs[510] +0.0126653*inputs[511] -0.04066*inputs[512] -0.0154642*inputs[513] -0.00265863*inputs[514] -0.0413284*inputs[515] -0.0185555*inputs[516] -0.00265865*inputs[517] -0.00319818*inputs[518] -0.0119086*inputs[519] -0.0221159*inputs[520] -0.0106477*inputs[521] -0.0104218*inputs[522] -0.0104218*inputs[523] -0.040805*inputs[524] +0.0218239*inputs[525] -0.0137109*inputs[526] -0.0117138*inputs[527] +0.00270548*inputs[528] -0.00265864*inputs[529] -0.0353636*inputs[530] +0.00820415*inputs[531] +0.00820417*inputs[532] -0.00265861*inputs[533] +0.0135744*inputs[534] +0.00787422*inputs[535] -0.0136796*inputs[536] -0.00596171*inputs[537] -0.00913608*inputs[538] +0.0131376*inputs[539] +0.0131376*inputs[540] -0.00276932*inputs[541] -0.00328854*inputs[542] +0.00144125*inputs[543] -0.0405352*inputs[544] -0.00265863*inputs[545] -0.00265862*inputs[546] +0.0048879*inputs[547] -0.0143527*inputs[548] -0.00265861*inputs[549] -0.00265861*inputs[550] -0.0349748*inputs[551] -0.020401*inputs[552] +0.00787422*inputs[553] -0.00440563*inputs[554] -0.00265866*inputs[555] +0.00576144*inputs[556] -0.0338273*inputs[557] -0.00397844*inputs[558] -0.0463934*inputs[559] -0.0029816*inputs[560] +0.025532*inputs[561] -0.000432582*inputs[562] -0.0390302*inputs[563] +0.0395867*inputs[564] +0.0121862*inputs[565] -0.00468906*inputs[566] -0.0238985*inputs[567] -0.00265862*inputs[568] +0.00924414*inputs[569] +0.00540032*inputs[570] -0.0274425*inputs[571] -0.0132282*inputs[572] +0.00533585*inputs[573] +0.00961273*inputs[574] -0.0169012*inputs[575] +0.00312361*inputs[576] -0.00264596*inputs[577] -0.00265862*inputs[578] -0.0235192*inputs[579] +0.0191101*inputs[580] -0.00187884*inputs[581] -0.00780815*inputs[582] -0.0151203*inputs[583] +0.000352227*inputs[584] -0.0278324*inputs[585] +0.00383078*inputs[586] +0.0147153*inputs[587] -0.00265862*inputs[588] -0.00265864*inputs[589] -0.0248177*inputs[590] +0.00193769*inputs[591] -0.0307712*inputs[592] -0.00466974*inputs[593] -0.011315*inputs[594] +0.00717029*inputs[595] +0.0225254*inputs[596] +0.000849677*inputs[597] -0.00426095*inputs[598] +0.0093422*inputs[599] +0.00901448*inputs[600] -0.00220203*inputs[601] -0.00265861*inputs[602] +0.0245124*inputs[603] +0.0202541*inputs[604] -0.0263575*inputs[605] -0.0306454*inputs[606] -0.00265865*inputs[607] -0.0400086*inputs[608] -0.012567*inputs[609] +0.00738883*inputs[610] +0.0161518*inputs[611] -0.0048764*inputs[612] +0.00146608*inputs[613] -0.00265866*inputs[614] -0.00265861*inputs[615] -0.0468731*inputs[616] +0.0155636*inputs[617] -0.00840512*inputs[618] -0.00468912*inputs[619] +0.00086831*inputs[620] -0.000263298*inputs[621] -0.0120478*inputs[622] +0.00248357*inputs[623] -0.0170133*inputs[624] +0.0124541*inputs[625] -0.00265861*inputs[626] -0.0129272*inputs[627] +0.0147153*inputs[628] -0.0233538*inputs[629] -0.0526672*inputs[630] +0.0268445*inputs[631] -0.00319818*inputs[632] -0.0021709*inputs[633] +0.00119251*inputs[634] +0.00521553*inputs[635] +0.00119251*inputs[636] +0.0253419*inputs[637] +0.0229815*inputs[638] +0.0130647*inputs[639] +0.0155653*inputs[640] -0.0176884*inputs[641] -0.00518052*inputs[642] -0.00265862*inputs[643] -0.00421857*inputs[644] -0.00840371*inputs[645] -0.0128258*inputs[646] +0.000831302*inputs[647] -0.00187884*inputs[648] -0.00400713*inputs[649] -0.0054404*inputs[650] -0.0117384*inputs[651] -0.0117384*inputs[652] -0.0117384*inputs[653] +0.00582255*inputs[654] -0.00187884*inputs[655] -0.00187885*inputs[656] -0.00187884*inputs[657] -0.00187885*inputs[658] -0.00187885*inputs[659] -0.00187892*inputs[660] -0.00187884*inputs[661] -0.00883048*inputs[662] -0.0110995*inputs[663] -0.0110995*inputs[664] -0.00187885*inputs[665] +0.0119895*inputs[666] -0.0185929*inputs[667] -0.0185929*inputs[668] -0.0305905*inputs[669] +0.0127479*inputs[670] -0.014624*inputs[671] -0.00187884*inputs[672] +0.00467174*inputs[673] +0.00467173*inputs[674] +0.0046717*inputs[675] -0.0118427*inputs[676] -0.0118427*inputs[677] -0.0118427*inputs[678] -0.0233412*inputs[679] +0.0227158*inputs[680] +0.0227159*inputs[681] +0.0227159*inputs[682] -0.00187885*inputs[683] -0.00187886*inputs[684] -0.00187884*inputs[685] -0.00187884*inputs[686] -0.00187885*inputs[687] -0.00943762*inputs[688] -0.00943771*inputs[689] -0.00187884*inputs[690] -0.00187884*inputs[691] -0.000400039*inputs[692] -0.000400086*inputs[693] -0.000400038*inputs[694] -0.000400039*inputs[695] -0.00187885*inputs[696] -0.00187888*inputs[697] -0.00187884*inputs[698] +0.00222149*inputs[699] +0.00222156*inputs[700] -0.0128521*inputs[701] -0.00460136*inputs[702] -0.00460127*inputs[703] -0.00460128*inputs[704] -0.00460128*inputs[705] +0.00607983*inputs[706] -0.00187885*inputs[707] -0.00187886*inputs[708] -0.00187891*inputs[709] -0.0018789*inputs[710] -0.00187885*inputs[711] -0.00187893*inputs[712] -0.0630804*inputs[713] -0.0233562*inputs[714] -0.0441936*inputs[715] +0.00729853*inputs[716] +0.00729853*inputs[717] -0.00187886*inputs[718] -0.00187887*inputs[719] -0.0108065*inputs[720] -0.0108065*inputs[721] +0.00126758*inputs[722] -0.00278614*inputs[723] -0.00187887*inputs[724] -0.00187884*inputs[725] -0.00187884*inputs[726] -0.00187884*inputs[727] -0.00187884*inputs[728] -0.0407501*inputs[729] +0.00312365*inputs[730] +0.00312367*inputs[731] +0.0031236*inputs[732] +0.00312367*inputs[733] +0.00312367*inputs[734] -0.00187884*inputs[735] -0.00187884*inputs[736] -0.00187884*inputs[737] +0.0162904*inputs[738] -0.00187885*inputs[739] -0.00187884*inputs[740] -0.00187884*inputs[741] +0.0147478*inputs[742] +0.0147478*inputs[743] +0.00491509*inputs[744] +0.028181*inputs[745] -0.000153779*inputs[746] -0.000153778*inputs[747] -0.0137029*inputs[748] -0.0137029*inputs[749] +0.00660985*inputs[750] +0.00660979*inputs[751] +0.0123233*inputs[752] -0.00187884*inputs[753] -0.0018789*inputs[754] +0.00481763*inputs[755] +0.00481764*inputs[756] +0.00481762*inputs[757] +0.00481764*inputs[758] +0.00959422*inputs[759] +0.00959422*inputs[760] -0.00187884*inputs[761] -0.00807391*inputs[762] -0.00807391*inputs[763] -0.00807393*inputs[764] -0.00807393*inputs[765] -0.00807393*inputs[766] -0.00807391*inputs[767] -0.0131191*inputs[768] -0.0131192*inputs[769] -0.00187887*inputs[770] +0.0119085*inputs[771] +0.0119085*inputs[772] -0.00187884*inputs[773] -0.0018789*inputs[774] -0.00187884*inputs[775] -0.00187885*inputs[776] -0.00187884*inputs[777] -0.00187884*inputs[778] -0.0018789*inputs[779] -0.00187886*inputs[780] -0.0442967*inputs[781] -0.0234829*inputs[782] -0.0234829*inputs[783] -0.00860903*inputs[784] -0.00860902*inputs[785] -0.0214709*inputs[786] -0.00187884*inputs[787] -0.00187884*inputs[788] -0.00187884*inputs[789] +0.0146396*inputs[790] -0.00187884*inputs[791] +0.000161196*inputs[792] -0.0018789*inputs[793] -0.00187884*inputs[794] +0.014912*inputs[795] +0.014912*inputs[796] +0.0226049*inputs[797] -0.00187888*inputs[798] -0.00187884*inputs[799] -0.00187884*inputs[800] +0.00185958*inputs[801] +0.0018596*inputs[802] -0.0130845*inputs[803] -0.00187884*inputs[804] -0.00670211*inputs[805] -0.00187886*inputs[806] +0.0072122*inputs[807] -0.00187885*inputs[808] -0.00187889*inputs[809] -0.00187884*inputs[810] -0.00187885*inputs[811] -0.0067021*inputs[812] +0.0404424*inputs[813] +0.0026063*inputs[814] +0.0026063*inputs[815] -0.00187889*inputs[816] +0.00721217*inputs[817] -0.00187886*inputs[818] -0.00187884*inputs[819] -0.00187887*inputs[820] -0.00187888*inputs[821] -0.032509*inputs[822] -0.0281893*inputs[823] -0.00468183*inputs[824] -0.00187884*inputs[825] -0.00187884*inputs[826] +0.0072122*inputs[827] -0.00468189*inputs[828] +0.0314049*inputs[829] +0.0130093*inputs[830] +0.00666803*inputs[831] +0.00666804*inputs[832] +0.0126259*inputs[833] -0.00468187*inputs[834] +0.00666806*inputs[835] +0.00669248*inputs[836] +0.0100229*inputs[837] -0.00187885*inputs[838] +0.0135995*inputs[839] -0.00187885*inputs[840] -0.000368286*inputs[841] +0.0193547*inputs[842] -0.00941884*inputs[843] -0.00941884*inputs[844] -0.00941884*inputs[845] -0.00941884*inputs[846] -0.00941884*inputs[847] +0.0236366*inputs[848] -0.0141132*inputs[849] -0.0141132*inputs[850] +0.0236366*inputs[851] -0.0239959*inputs[852] -0.0131715*inputs[853] +0.0135428*inputs[854] +0.00886437*inputs[855] -0.00187886*inputs[856] -0.00187885*inputs[857] -0.00187886*inputs[858] +0.00245334*inputs[859] -0.00187884*inputs[860] -0.00187888*inputs[861] -0.00187891*inputs[862] -0.00187884*inputs[863] -0.00187885*inputs[864] +0.0231563*inputs[865] -0.00374448*inputs[866] -0.0131715*inputs[867] -0.0161196*inputs[868] -0.00187885*inputs[869] -0.00846939*inputs[870] -0.00846939*inputs[871] -0.00187885*inputs[872] -0.00770601*inputs[873] -0.00770604*inputs[874] -0.00770601*inputs[875] -0.00187888*inputs[876] +0.0066925*inputs[877] -0.0223696*inputs[878] -0.0243246*inputs[879] -0.0223695*inputs[880] +0.00814305*inputs[881] -0.0367767*inputs[882] -0.00187884*inputs[883] -0.00187887*inputs[884] +0.00335696*inputs[885] +0.00335695*inputs[886] -0.0018789*inputs[887] -0.00187884*inputs[888] -0.00187884*inputs[889] -0.00187889*inputs[890] -0.00187885*inputs[891] -0.00187884*inputs[892] -0.00187889*inputs[893] -0.00187884*inputs[894] -0.00187892*inputs[895] -0.00187884*inputs[896] -0.00726738*inputs[897] -0.00187888*inputs[898] -0.00179066*inputs[899] -0.00726746*inputs[900] +0.00669249*inputs[901] +0.00779949*inputs[902] -0.0192311*inputs[903] -0.00187886*inputs[904] +0.00779949*inputs[905] +0.00779949*inputs[906] -0.00127118*inputs[907] -0.00127119*inputs[908] -0.00187888*inputs[909] -0.00187884*inputs[910] -0.00187884*inputs[911] +0.0310171*inputs[912] -0.0221693*inputs[913] -0.00187884*inputs[914] -0.00187888*inputs[915] -0.0192311*inputs[916] -0.00165685*inputs[917] -0.00187885*inputs[918] -0.00165685*inputs[919] -0.00165692*inputs[920] -0.0192311*inputs[921] -0.0374619*inputs[922] +0.0133722*inputs[923] -0.0128869*inputs[924] -0.00187888*inputs[925] -0.00187884*inputs[926] -0.00187885*inputs[927] -0.00187886*inputs[928] -0.0329418*inputs[929] -0.00187887*inputs[930] -0.00187887*inputs[931] -0.0306978*inputs[932] -0.0123792*inputs[933] +0.00742164*inputs[934] -0.00187889*inputs[935] -0.0123792*inputs[936] -0.0083951*inputs[937] +0.00742172*inputs[938] -0.0162679*inputs[939] -0.00412001*inputs[940] -0.00412*inputs[941] -0.00412006*inputs[942] -0.00412006*inputs[943] +0.000909501*inputs[944] -0.0185929*inputs[945] +0.000909535*inputs[946] +0.0238888*inputs[947] -0.00187885*inputs[948] +0.00729391*inputs[949] +0.00729391*inputs[950] -0.0185929*inputs[951] -0.0214443*inputs[952] -0.00187892*inputs[953] -0.00187889*inputs[954] +0.0108649*inputs[955] +0.0108649*inputs[956] +0.0108649*inputs[957] +0.0108649*inputs[958] -0.0195775*inputs[959] -0.00187885*inputs[960] -0.00187884*inputs[961] +0.00999637*inputs[962] +0.00999638*inputs[963] +0.00353933*inputs[964] +0.0291714*inputs[965] -0.0018789*inputs[966] -0.00187884*inputs[967] -0.00187892*inputs[968] -0.0325085*inputs[969] -0.00187884*inputs[970] +0.0186021*inputs[971] -0.00187886*inputs[972] -0.0018789*inputs[973] -0.00187884*inputs[974] -0.00187894*inputs[975] -0.0505689*inputs[976] -0.0186863*inputs[977] +0.0132852*inputs[978] -0.00187885*inputs[979] -0.0204788*inputs[980] -0.00187884*inputs[981] +0.0264453*inputs[982] +0.0264453*inputs[983] -0.00187886*inputs[984] -0.00187884*inputs[985] -0.00187884*inputs[986] +0.000144207*inputs[987] -0.00187884*inputs[988] -0.00187884*inputs[989] -0.00187885*inputs[990] -0.00785554*inputs[991] -0.0163552*inputs[992] -0.00187891*inputs[993] +0.00381515*inputs[994] -0.00187884*inputs[995] -0.019977*inputs[996] +0.0306949*inputs[997] -0.00785555*inputs[998] -0.00187889*inputs[999] -0.00187885*inputs[1000] -0.00187893*inputs[1001] -0.0108237*inputs[1002] -0.00414384*inputs[1003] -0.00785554*inputs[1004] -0.00187891*inputs[1005] -0.00187886*inputs[1006] -0.00187884*inputs[1007] -0.0237332*inputs[1008] +0.0120143*inputs[1009] +0.0120143*inputs[1010] +0.0120143*inputs[1011] -0.00187884*inputs[1012] -0.00187886*inputs[1013] -0.00187884*inputs[1014] -0.00187886*inputs[1015] -0.0104786*inputs[1016] -0.0104786*inputs[1017] +0.0150843*inputs[1018] +0.0150843*inputs[1019] -0.025097*inputs[1020] -0.00123346*inputs[1021] -0.00187884*inputs[1022] -0.00187889*inputs[1023] -0.00785558*inputs[1024] -0.00383981*inputs[1025] -0.00383981*inputs[1026] -0.00383981*inputs[1027] -0.00187884*inputs[1028] -0.00187884*inputs[1029] -0.00187884*inputs[1030] -0.00785556*inputs[1031] +0.0057033*inputs[1032] +0.0136377*inputs[1033] +0.00570324*inputs[1034] -0.0161077*inputs[1035] -0.0161077*inputs[1036] -0.0161077*inputs[1037] -0.0161077*inputs[1038] -0.0161077*inputs[1039] -0.00187887*inputs[1040] -0.00187885*inputs[1041] -0.00187892*inputs[1042] +0.0109305*inputs[1043] -0.0170528*inputs[1044] -0.0170528*inputs[1045] -0.0170528*inputs[1046] +0.0109305*inputs[1047] +0.0102111*inputs[1048] -0.00187884*inputs[1049] -0.0171005*inputs[1050] +0.0102111*inputs[1051] +0.0109305*inputs[1052] -0.00187884*inputs[1053] +0.0039514*inputs[1054] +0.00395141*inputs[1055] +0.00395137*inputs[1056] +0.00395141*inputs[1057] -0.0149538*inputs[1058] +0.0323066*inputs[1059] -0.00187884*inputs[1060] -0.00187886*inputs[1061] -0.00187885*inputs[1062] +0.0149458*inputs[1063] +0.0328174*inputs[1064] -0.00187885*inputs[1065] -0.0110348*inputs[1066] -0.0110348*inputs[1067] -0.00187887*inputs[1068] -0.00187889*inputs[1069] -0.00187891*inputs[1070] -0.00187888*inputs[1071] -0.00187885*inputs[1072] -0.00187884*inputs[1073] -0.00187886*inputs[1074] -0.00187893*inputs[1075] -0.00187885*inputs[1076] -0.00187884*inputs[1077] +0.00506042*inputs[1078] +0.0238782*inputs[1079] +0.0180442*inputs[1080] +0.00506045*inputs[1081] +0.00506044*inputs[1082] -0.00474897*inputs[1083] -0.00474899*inputs[1084] -0.00474901*inputs[1085] -0.00187888*inputs[1086] -0.00187884*inputs[1087] +0.00506043*inputs[1088] -0.039573*inputs[1089] -0.00187884*inputs[1090] +0.0102111*inputs[1091] -0.0220289*inputs[1092] -0.0018789*inputs[1093] +0.0100705*inputs[1094] -0.00187886*inputs[1095] +0.0102111*inputs[1096] -0.00187891*inputs[1097] -0.00187884*inputs[1098] +7.68961e-05*inputs[1099] +7.69145e-05*inputs[1100] -0.00187888*inputs[1101] -0.00187885*inputs[1102] +0.0127574*inputs[1103] +0.0210397*inputs[1104] -0.00187884*inputs[1105] -0.00187893*inputs[1106] -0.00187887*inputs[1107] -0.00187886*inputs[1108] -0.0148023*inputs[1109] +0.00391627*inputs[1110] +0.0289158*inputs[1111] -0.0163936*inputs[1112] -0.00187884*inputs[1113] -0.0151779*inputs[1114] -0.00187884*inputs[1115] -0.0592478*inputs[1116] -0.0298498*inputs[1117] +0.0150996*inputs[1118] -0.00187892*inputs[1119] -0.00187888*inputs[1120] -0.00187884*inputs[1121] -0.00187884*inputs[1122] -0.00187884*inputs[1123] -0.00187885*inputs[1124] -0.00187884*inputs[1125] -0.00620553*inputs[1126] -0.00620553*inputs[1127] -0.00187884*inputs[1128] -0.00187885*inputs[1129] -0.0263957*inputs[1130] -0.036911*inputs[1131] +0.0266125*inputs[1132] -0.00851052*inputs[1133] -0.00851053*inputs[1134] -0.0111143*inputs[1135] -0.0111143*inputs[1136] -0.0111143*inputs[1137] -0.0152622*inputs[1138] -0.00187884*inputs[1139] -0.00187884*inputs[1140] +0.00557137*inputs[1141] -0.0151779*inputs[1142] -0.00264155*inputs[1143] -0.00264153*inputs[1144] -0.00264157*inputs[1145] -0.00264153*inputs[1146] +0.0262373*inputs[1147] +0.0262373*inputs[1148] +0.0112009*inputs[1149] -0.00187888*inputs[1150] -0.00187884*inputs[1151] -0.00187885*inputs[1152] +0.00709175*inputs[1153] +0.00709174*inputs[1154] +0.00709174*inputs[1155] -0.0147736*inputs[1156] -0.0213319*inputs[1157] -0.0322563*inputs[1158] -0.00187884*inputs[1159] -0.00187884*inputs[1160] +0.00535294*inputs[1161] -0.00187884*inputs[1162] +0.0221597*inputs[1163] -0.00187884*inputs[1164] -0.00187893*inputs[1165] -0.00187884*inputs[1166] -0.023989*inputs[1167] -0.00187893*inputs[1168] -0.00187885*inputs[1169] -0.00187885*inputs[1170] -0.00187885*inputs[1171] -0.00187888*inputs[1172] +0.00356469*inputs[1173] +0.0107287*inputs[1174] +0.0107287*inputs[1175] +0.0107288*inputs[1176] +0.0107288*inputs[1177] +0.0120319*inputs[1178] +0.0120319*inputs[1179] +0.0120319*inputs[1180] +0.0120319*inputs[1181] -0.00187884*inputs[1182] -0.00187887*inputs[1183] -0.00187884*inputs[1184] -0.00187884*inputs[1185] -0.0334413*inputs[1186] -0.00187887*inputs[1187] -0.00187884*inputs[1188] -0.00187885*inputs[1189] +0.0284789*inputs[1190] -0.00187885*inputs[1191] +0.00310652*inputs[1192] -0.0324583*inputs[1193] +0.0265945*inputs[1194] -0.00187884*inputs[1195] -0.0018789*inputs[1196] +0.0288531*inputs[1197] +0.0203461*inputs[1198] +0.0203461*inputs[1199] -0.0238192*inputs[1200] -0.00187885*inputs[1201] -0.00187891*inputs[1202] -0.00187884*inputs[1203] -0.00187884*inputs[1204] -0.0224427*inputs[1205] +0.00775799*inputs[1206] +0.00775793*inputs[1207] -0.00187888*inputs[1208] -0.00187885*inputs[1209] -0.00187891*inputs[1210] -0.00187885*inputs[1211] -0.00187885*inputs[1212] -0.00187884*inputs[1213] -0.00187884*inputs[1214] -0.00187885*inputs[1215] -0.00187885*inputs[1216] -0.00187884*inputs[1217] -0.00187884*inputs[1218] +0.0197377*inputs[1219] +0.0197377*inputs[1220] +0.0197377*inputs[1221] -0.0125081*inputs[1222] -0.0125082*inputs[1223] -0.0125081*inputs[1224] -0.019361*inputs[1225] -0.019361*inputs[1226] -0.00187893*inputs[1227] -0.00187884*inputs[1228] -0.00187885*inputs[1229] -0.00590646*inputs[1230] -0.00590642*inputs[1231] -0.0325088*inputs[1232] -0.00187885*inputs[1233] -0.00187884*inputs[1234] -0.00187885*inputs[1235] +0.0301226*inputs[1236] -0.00187884*inputs[1237] -0.0184086*inputs[1238] -0.0180962*inputs[1239] -0.0180961*inputs[1240] -0.00187885*inputs[1241] -0.00187884*inputs[1242] -0.00187884*inputs[1243] -0.00187885*inputs[1244] -0.00187886*inputs[1245] -0.00764226*inputs[1246] -0.00764226*inputs[1247] -0.00764226*inputs[1248] +0.0133616*inputs[1249] -0.00187884*inputs[1250] -0.00187885*inputs[1251] -0.00740459*inputs[1252] -0.00187884*inputs[1253] -0.0158843*inputs[1254] -0.0047216*inputs[1255] -0.0047216*inputs[1256] -0.0179023*inputs[1257] -0.00999954*inputs[1258] -0.00999958*inputs[1259] +0.0030541*inputs[1260] +0.00414033*inputs[1261] +0.0030541*inputs[1262] +0.0030541*inputs[1263] +0.0030541*inputs[1264] -0.0322052*inputs[1265] +0.0385754*inputs[1266] -0.00187885*inputs[1267] -0.00187884*inputs[1268] -0.00187884*inputs[1269] +0.0212831*inputs[1270] +0.020375*inputs[1271] -0.00713164*inputs[1272] -0.00713167*inputs[1273] +0.010643*inputs[1274] +0.010643*inputs[1275] -0.00187884*inputs[1276] -0.00187885*inputs[1277] +0.0211469*inputs[1278] -0.00187884*inputs[1279] -0.00187888*inputs[1280] +0.00414026*inputs[1281] -0.00187886*inputs[1282] -0.00187884*inputs[1283] -0.0295532*inputs[1284] -0.0131827*inputs[1285] -0.0470644*inputs[1286] -0.00187888*inputs[1287] -0.0018789*inputs[1288] -0.00187889*inputs[1289] -0.00545119*inputs[1290] -0.0114925*inputs[1291] -0.0162503*inputs[1292] -0.0299688*inputs[1293] -0.0125373*inputs[1294] -0.0067321*inputs[1295] -0.00187892*inputs[1296] -0.0114925*inputs[1297] +0.0307282*inputs[1298] -0.00187884*inputs[1299] -0.00187885*inputs[1300] -0.00187884*inputs[1301] -0.00187884*inputs[1302] +0.00481386*inputs[1303] -0.0018789*inputs[1304] +0.0105075*inputs[1305] +0.0105075*inputs[1306] +0.0105075*inputs[1307] -0.00187884*inputs[1308] +0.00859264*inputs[1309] -0.0018789*inputs[1310] -0.00187884*inputs[1311] -0.00187884*inputs[1312] -0.00701869*inputs[1313] -0.00701869*inputs[1314] -0.00187885*inputs[1315] -0.00187884*inputs[1316] -0.00553326*inputs[1317] -0.00187884*inputs[1318] -0.0165814*inputs[1319] +0.0312716*inputs[1320] -0.00187885*inputs[1321] -0.00655985*inputs[1322] -0.00187884*inputs[1323] -0.00187889*inputs[1324] +0.0386653*inputs[1325] +0.00878812*inputs[1326] -0.00187884*inputs[1327] -0.00187887*inputs[1328] -0.00187889*inputs[1329] -0.00187884*inputs[1330] -0.0141254*inputs[1331] -0.00187884*inputs[1332] -0.00187886*inputs[1333] -0.00187885*inputs[1334] -0.00187884*inputs[1335] +0.0127895*inputs[1336] +0.0127895*inputs[1337] +0.0127894*inputs[1338] -0.0113973*inputs[1339] -0.0113972*inputs[1340] -0.00187887*inputs[1341] -0.00807724*inputs[1342] -0.00187884*inputs[1343] -0.00187885*inputs[1344] -0.00187891*inputs[1345] -0.016819*inputs[1346] -0.00187884*inputs[1347] -0.00187884*inputs[1348] -0.00187886*inputs[1349] -0.00187884*inputs[1350] -0.00187884*inputs[1351] -0.0018789*inputs[1352] +0.0120142*inputs[1353] +0.0120143*inputs[1354] -0.00187884*inputs[1355] -0.00187884*inputs[1356] +0.0328179*inputs[1357] +0.00250454*inputs[1358] -0.00187885*inputs[1359] -0.00187884*inputs[1360] -0.00187884*inputs[1361] -0.00187885*inputs[1362] -0.00187884*inputs[1363] +0.0219351*inputs[1364] +0.0219351*inputs[1365] -8.75629e-05*inputs[1366] -8.75657e-05*inputs[1367] -8.75642e-05*inputs[1368] -8.75763e-05*inputs[1369] -8.75743e-05*inputs[1370] -0.0218954*inputs[1371] -0.00650732*inputs[1372] -0.00650725*inputs[1373] +0.00769772*inputs[1374] -0.00660752*inputs[1375] -0.00660752*inputs[1376] -0.0141743*inputs[1377] -0.0498209*inputs[1378] +0.00157387*inputs[1379] -0.00187884*inputs[1380] -0.0018789*inputs[1381] -0.0018789*inputs[1382] -0.00187887*inputs[1383] -0.0342482*inputs[1384] -0.0163387*inputs[1385] -0.0163387*inputs[1386] -0.00187887*inputs[1387] -0.00187884*inputs[1388] -0.00187889*inputs[1389] -0.030015*inputs[1390] -0.0018789*inputs[1391] -0.00187886*inputs[1392] -0.00187885*inputs[1393] -0.00187888*inputs[1394] -0.00187886*inputs[1395] 
		combinations[3] = 0.0484227 +0.0776572*inputs[0] -0.0164614*inputs[1] +0.221423*inputs[2] +0.159753*inputs[3] +0.149545*inputs[4] +0.069072*inputs[5] +0.0105009*inputs[6] +0.034091*inputs[7] +0.0475994*inputs[8] -0.00807528*inputs[9] +0.0253451*inputs[10] +0.0948953*inputs[11] +0.028984*inputs[12] +0.098573*inputs[13] +0.176009*inputs[14] +0.0346916*inputs[15] +0.00679073*inputs[16] +0.0119289*inputs[17] +0.105998*inputs[18] -0.0975062*inputs[19] +0.00478507*inputs[20] +0.0154814*inputs[21] +0.0264019*inputs[22] +0.021644*inputs[23] -0.00379527*inputs[24] -0.0233334*inputs[25] +0.12523*inputs[26] -0.0128846*inputs[27] +0.0118402*inputs[28] +0.00211251*inputs[29] +0.0138656*inputs[30] -0.021448*inputs[31] -0.10939*inputs[32] +0.0963665*inputs[33] -0.0803933*inputs[34] +0.102769*inputs[35] +0.0901933*inputs[36] +0.0090792*inputs[37] -0.00769655*inputs[38] -0.125018*inputs[39] +0.0401873*inputs[40] -0.118151*inputs[41] +0.013473*inputs[42] -0.0412377*inputs[43] +0.0342272*inputs[44] +0.0115224*inputs[45] -0.0196763*inputs[46] -0.106382*inputs[47] -0.0787325*inputs[48] +0.0889176*inputs[49] +0.00273101*inputs[50] +0.00318274*inputs[51] +0.00143262*inputs[52] +0.0102194*inputs[53] -0.00332667*inputs[54] -0.0469478*inputs[55] +0.0568024*inputs[56] +0.0326243*inputs[57] -0.104905*inputs[58] -0.00387601*inputs[59] -0.120214*inputs[60] -0.0302242*inputs[61] -0.0289951*inputs[62] -0.035401*inputs[63] +0.00177642*inputs[64] -0.150931*inputs[65] -0.0809434*inputs[66] +0.046937*inputs[67] -0.0396268*inputs[68] +0.000926538*inputs[69] +0.0111886*inputs[70] -0.00260269*inputs[71] +0.0103884*inputs[72] -0.0110252*inputs[73] -0.090244*inputs[74] +0.00877102*inputs[75] +0.0984228*inputs[76] +0.0177283*inputs[77] +0.0171225*inputs[78] +0.0109428*inputs[79] -0.0410245*inputs[80] -0.0396791*inputs[81] +0.0101485*inputs[82] +0.0829591*inputs[83] +0.0239287*inputs[84] -0.0405285*inputs[85] +0.0431334*inputs[86] -0.0172423*inputs[87] -0.00448414*inputs[88] -0.0153149*inputs[89] -0.0106062*inputs[90] -0.0231778*inputs[91] -0.00263546*inputs[92] -0.0219964*inputs[93] -0.105947*inputs[94] -0.0618016*inputs[95] +0.00687046*inputs[96] +0.0211569*inputs[97] -0.0123893*inputs[98] +0.0368055*inputs[99] +0.00669537*inputs[100] -0.0272742*inputs[101] -0.0213942*inputs[102] -0.0127986*inputs[103] -0.0466151*inputs[104] +0.00306121*inputs[105] +0.202*inputs[106] +0.00558935*inputs[107] -0.0818986*inputs[108] +0.00709773*inputs[109] +0.0448759*inputs[110] +0.0584144*inputs[111] -0.0197251*inputs[112] +0.0185902*inputs[113] -0.0618758*inputs[114] +0.0197902*inputs[115] -0.0037389*inputs[116] -0.0164661*inputs[117] -0.0322201*inputs[118] -0.0140979*inputs[119] -0.0275644*inputs[120] +0.0792283*inputs[121] +0.0723672*inputs[122] -0.0803996*inputs[123] -0.027745*inputs[124] -0.0122758*inputs[125] -0.00366475*inputs[126] -0.0035736*inputs[127] +0.009282*inputs[128] +0.00714291*inputs[129] +0.0116954*inputs[130] +0.00881128*inputs[131] +0.020493*inputs[132] +0.0116222*inputs[133] -0.0317571*inputs[134] -0.0979797*inputs[135] -0.0732234*inputs[136] -0.00986429*inputs[137] -0.0223204*inputs[138] -0.0833601*inputs[139] +0.00826576*inputs[140] +0.0216678*inputs[141] +0.0282547*inputs[142] +0.0381008*inputs[143] -0.0143551*inputs[144] +0.00873107*inputs[145] +0.064006*inputs[146] -0.0198541*inputs[147] -0.0251179*inputs[148] -0.0370724*inputs[149] -0.016729*inputs[150] -0.0144075*inputs[151] -0.0367965*inputs[152] -0.000897294*inputs[153] -0.0588308*inputs[154] -0.0156928*inputs[155] +0.00323328*inputs[156] -0.00814621*inputs[157] -0.0205241*inputs[158] +0.0630387*inputs[159] -0.00352516*inputs[160] -0.0139712*inputs[161] +0.00204259*inputs[162] -0.0220397*inputs[163] -0.00769596*inputs[164] -0.00364368*inputs[165] +0.0193429*inputs[166] +0.00605911*inputs[167] +0.00230456*inputs[168] -0.0216019*inputs[169] -0.0122465*inputs[170] -0.0363186*inputs[171] +0.00707162*inputs[172] +0.0202831*inputs[173] -0.018219*inputs[174] +0.0632038*inputs[175] +0.0145027*inputs[176] -0.00750145*inputs[177] -0.0684165*inputs[178] +0.0462263*inputs[179] -0.0562029*inputs[180] -0.0149692*inputs[181] +0.00368704*inputs[182] +0.00567387*inputs[183] -0.0104915*inputs[184] -0.00102195*inputs[185] -0.0118134*inputs[186] -0.0114374*inputs[187] +0.0150255*inputs[188] -0.009068*inputs[189] -0.0233463*inputs[190] -0.0115531*inputs[191] +0.0339171*inputs[192] -0.00366869*inputs[193] +0.0488638*inputs[194] +0.0184537*inputs[195] +0.0191613*inputs[196] -0.0358936*inputs[197] +0.0347781*inputs[198] -0.00739235*inputs[199] -0.0032704*inputs[200] +0.00466306*inputs[201] -0.00411809*inputs[202] -0.0125473*inputs[203] +0.0265965*inputs[204] -0.0403112*inputs[205] +0.0215418*inputs[206] -0.0251746*inputs[207] -0.0174909*inputs[208] -0.0530886*inputs[209] +0.00103958*inputs[210] -0.00588081*inputs[211] -0.0275664*inputs[212] -0.00306953*inputs[213] +0.016811*inputs[214] +0.0550768*inputs[215] +0.0260876*inputs[216] -0.0215088*inputs[217] +0.0197932*inputs[218] -0.0069847*inputs[219] +0.0167643*inputs[220] -0.0187132*inputs[221] +0.0138135*inputs[222] +0.00262188*inputs[223] -0.000333036*inputs[224] -0.053315*inputs[225] -0.000180801*inputs[226] -0.00759641*inputs[227] +0.00351413*inputs[228] +0.0151712*inputs[229] +0.0195936*inputs[230] +0.00442288*inputs[231] +0.0150944*inputs[232] +0.00960248*inputs[233] +0.00176213*inputs[234] +0.0400983*inputs[235] -0.000731634*inputs[236] +0.0220873*inputs[237] -0.0113773*inputs[238] -0.0400949*inputs[239] -0.00342972*inputs[240] -0.00180192*inputs[241] -0.00499522*inputs[242] -0.0169247*inputs[243] -0.0136865*inputs[244] +0.0363877*inputs[245] -0.0131993*inputs[246] -0.00410796*inputs[247] -0.046877*inputs[248] -0.00773461*inputs[249] -0.00895457*inputs[250] -0.0280992*inputs[251] -0.0116135*inputs[252] -0.0353625*inputs[253] -0.012143*inputs[254] -0.00764859*inputs[255] +0.0112229*inputs[256] -0.00278788*inputs[257] +0.00101551*inputs[258] -0.00636239*inputs[259] -0.00306601*inputs[260] +0.0290818*inputs[261] +0.0118822*inputs[262] -0.000906827*inputs[263] +0.00439395*inputs[264] -0.0103814*inputs[265] +0.0225018*inputs[266] -0.0578783*inputs[267] -0.0233308*inputs[268] +0.00961358*inputs[269] -0.00861962*inputs[270] -0.0524863*inputs[271] -0.00483489*inputs[272] +0.00693526*inputs[273] -0.00306601*inputs[274] -0.00306601*inputs[275] +0.00996459*inputs[276] -0.0035632*inputs[277] -0.0497851*inputs[278] +0.00233809*inputs[279] +0.0226749*inputs[280] -0.00240672*inputs[281] +0.0198248*inputs[282] +0.0297664*inputs[283] +0.0201286*inputs[284] +0.100938*inputs[285] -0.0116572*inputs[286] -0.0284662*inputs[287] -0.0162886*inputs[288] +0.0192971*inputs[289] +0.0256116*inputs[290] -0.00184771*inputs[291] -0.00771361*inputs[292] -0.0164228*inputs[293] +0.0179847*inputs[294] +0.00421633*inputs[295] -0.0157741*inputs[296] +0.0010577*inputs[297] +0.0199948*inputs[298] +0.0471629*inputs[299] +0.0312649*inputs[300] +0.00931333*inputs[301] +0.0052935*inputs[302] -0.0851573*inputs[303] +0.0056752*inputs[304] -0.0427395*inputs[305] -0.018065*inputs[306] -0.0383214*inputs[307] -0.0172035*inputs[308] -0.020296*inputs[309] -0.00278788*inputs[310] -0.00865213*inputs[311] -0.0489117*inputs[312] +0.0111275*inputs[313] +0.0386219*inputs[314] +0.00347634*inputs[315] -0.00669903*inputs[316] +0.00943663*inputs[317] +0.00134437*inputs[318] +0.0271996*inputs[319] -0.0281727*inputs[320] -0.00676061*inputs[321] -0.00553444*inputs[322] -0.0160891*inputs[323] -0.00312267*inputs[324] -0.000358169*inputs[325] -0.00628541*inputs[326] +0.00416254*inputs[327] -0.000726533*inputs[328] -0.00265383*inputs[329] -0.0130005*inputs[330] +0.0178053*inputs[331] +0.028489*inputs[332] -0.00553445*inputs[333] -0.0255477*inputs[334] -0.0228594*inputs[335] -0.0112224*inputs[336] +0.0179004*inputs[337] -0.0411806*inputs[338] -0.00631321*inputs[339] -0.0333188*inputs[340] +0.0471832*inputs[341] +0.000195672*inputs[342] +0.010073*inputs[343] -0.0204289*inputs[344] +0.0286044*inputs[345] -0.016541*inputs[346] -0.0020543*inputs[347] -0.0189822*inputs[348] +0.00671085*inputs[349] -0.000284712*inputs[350] +0.0387866*inputs[351] -0.00663412*inputs[352] +0.00986088*inputs[353] -0.0138814*inputs[354] +0.00864239*inputs[355] +0.00383279*inputs[356] +0.0107826*inputs[357] -0.00265383*inputs[358] +0.00747478*inputs[359] +0.0140809*inputs[360] +0.000149712*inputs[361] -0.00295186*inputs[362] -0.0126217*inputs[363] +0.0122214*inputs[364] -0.0172019*inputs[365] +0.000694023*inputs[366] -0.021888*inputs[367] +0.0192685*inputs[368] -0.0385387*inputs[369] -0.00265385*inputs[370] -0.00265383*inputs[371] -0.00913133*inputs[372] -0.0119248*inputs[373] +0.0179876*inputs[374] -0.010031*inputs[375] +0.00909927*inputs[376] +0.0345399*inputs[377] +0.0117254*inputs[378] +0.0171022*inputs[379] +0.0027495*inputs[380] +0.0411096*inputs[381] -0.0229363*inputs[382] -0.0111188*inputs[383] -0.0743812*inputs[384] -0.00535756*inputs[385] -0.00541972*inputs[386] +0.0095074*inputs[387] -0.00265388*inputs[388] -0.0122297*inputs[389] -0.00815847*inputs[390] +0.00789466*inputs[391] -0.00265383*inputs[392] +0.0297675*inputs[393] +0.0149309*inputs[394] -0.0411552*inputs[395] +0.00354409*inputs[396] +0.0102919*inputs[397] +0.013674*inputs[398] -0.0113635*inputs[399] +0.0234927*inputs[400] +0.0275713*inputs[401] -0.0108253*inputs[402] +0.0155384*inputs[403] -0.00645826*inputs[404] +0.000131136*inputs[405] -0.0629234*inputs[406] +0.000193254*inputs[407] -0.0278694*inputs[408] -0.0114257*inputs[409] +0.000776575*inputs[410] -0.044747*inputs[411] +0.00384506*inputs[412] -0.00265383*inputs[413] -0.0197361*inputs[414] +0.00389267*inputs[415] -0.0243542*inputs[416] +0.000764798*inputs[417] -0.0173185*inputs[418] -0.0116732*inputs[419] -0.00072247*inputs[420] -0.0300528*inputs[421] -0.00265384*inputs[422] -0.0245886*inputs[423] -0.0137869*inputs[424] -0.0183644*inputs[425] -0.0105233*inputs[426] -0.0351972*inputs[427] +0.0118463*inputs[428] +0.00852171*inputs[429] +0.00925314*inputs[430] +0.0137272*inputs[431] -0.0470137*inputs[432] +0.0107127*inputs[433] -0.0188376*inputs[434] -0.00909052*inputs[435] -0.00545893*inputs[436] +0.00330007*inputs[437] +0.000884245*inputs[438] -0.0310061*inputs[439] -0.000173865*inputs[440] -0.00265386*inputs[441] -0.0214426*inputs[442] -0.0157292*inputs[443] -0.00666115*inputs[444] -0.0333545*inputs[445] -0.0740994*inputs[446] +0.0255808*inputs[447] -0.0178338*inputs[448] -0.00632482*inputs[449] -0.00216568*inputs[450] -0.0380214*inputs[451] -0.0120823*inputs[452] -0.00967403*inputs[453] +0.0160436*inputs[454] -0.0244737*inputs[455] -0.0244737*inputs[456] +0.00444669*inputs[457] +0.00400572*inputs[458] +0.054366*inputs[459] -0.00216568*inputs[460] -0.0251291*inputs[461] -0.0229625*inputs[462] -0.00153049*inputs[463] -0.0118661*inputs[464] -0.0021657*inputs[465] +0.0157479*inputs[466] -0.0579791*inputs[467] +0.00742154*inputs[468] +0.00801286*inputs[469] +0.00143049*inputs[470] +0.020363*inputs[471] +0.0339498*inputs[472] -0.0134073*inputs[473] +0.00859006*inputs[474] +0.0137368*inputs[475] -0.0145747*inputs[476] -0.0171178*inputs[477] +0.0064394*inputs[478] -0.0268999*inputs[479] -0.0228419*inputs[480] +0.0223014*inputs[481] -0.0042011*inputs[482] -0.0116823*inputs[483] -0.00216569*inputs[484] -0.00305009*inputs[485] -0.00916438*inputs[486] -0.00216568*inputs[487] +0.0244923*inputs[488] -0.0126805*inputs[489] -0.0899913*inputs[490] -0.00215238*inputs[491] +0.00682963*inputs[492] -0.0216322*inputs[493] +0.00318736*inputs[494] +0.00534781*inputs[495] -0.025247*inputs[496] -0.0294548*inputs[497] -0.0168755*inputs[498] +0.00368124*inputs[499] +0.0395798*inputs[500] +0.0184826*inputs[501] -0.0233658*inputs[502] -0.00216569*inputs[503] -0.0302001*inputs[504] -0.00216568*inputs[505] +0.00577482*inputs[506] +0.0279887*inputs[507] +0.0078374*inputs[508] -0.0252392*inputs[509] +0.0172384*inputs[510] +0.0146493*inputs[511] -0.0392218*inputs[512] -0.0154676*inputs[513] -0.00216569*inputs[514] -0.0386047*inputs[515] -0.017356*inputs[516] -0.00216568*inputs[517] -0.00294685*inputs[518] -0.0123757*inputs[519] -0.0193016*inputs[520] -0.010095*inputs[521] -0.00994508*inputs[522] -0.00994508*inputs[523] -0.0397916*inputs[524] +0.0210873*inputs[525] -0.0131161*inputs[526] -0.010263*inputs[527] +0.00362583*inputs[528] -0.00216569*inputs[529] -0.00710088*inputs[530] +0.00860183*inputs[531] +0.00860183*inputs[532] -0.00216569*inputs[533] +0.0136816*inputs[534] +0.0102331*inputs[535] -0.0109441*inputs[536] -0.0062739*inputs[537] -0.00890405*inputs[538] +0.0160517*inputs[539] +0.0160517*inputs[540] -0.00229367*inputs[541] -0.00270559*inputs[542] +0.00171138*inputs[543] -0.038912*inputs[544] -0.00216574*inputs[545] -0.00216568*inputs[546] +0.00500435*inputs[547] -0.0115222*inputs[548] -0.00216568*inputs[549] -0.00216568*inputs[550] -0.031196*inputs[551] -0.0210004*inputs[552] +0.010233*inputs[553] -0.0055708*inputs[554] -0.00216573*inputs[555] +0.00434584*inputs[556] -0.0330286*inputs[557] -0.00350928*inputs[558] -0.0449203*inputs[559] -0.00413021*inputs[560] +0.023308*inputs[561] -0.000563274*inputs[562] -0.0398358*inputs[563] +0.0388677*inputs[564] +0.0128339*inputs[565] -0.00433262*inputs[566] -0.0230767*inputs[567] -0.00216568*inputs[568] +0.00917972*inputs[569] +0.00569264*inputs[570] -0.0267664*inputs[571] -0.0127462*inputs[572] +0.00603688*inputs[573] +0.00945487*inputs[574] -0.0137901*inputs[575] +0.00217244*inputs[576] -0.00215234*inputs[577] -0.00216568*inputs[578] -0.0236919*inputs[579] +0.0186633*inputs[580] -0.00153057*inputs[581] -0.00622562*inputs[582] -0.0143677*inputs[583] +0.00580198*inputs[584] -0.0273049*inputs[585] +0.00444875*inputs[586] +0.0134062*inputs[587] -0.0021657*inputs[588] -0.00216571*inputs[589] -0.0240958*inputs[590] +0.00783738*inputs[591] -0.0315244*inputs[592] -0.00492757*inputs[593] -0.0100521*inputs[594] +0.00826601*inputs[595] +0.0219002*inputs[596] +0.00120636*inputs[597] -0.00513992*inputs[598] +0.0100628*inputs[599] -0.000332186*inputs[600] -0.00163028*inputs[601] -0.00216571*inputs[602] +0.0241864*inputs[603] +0.0200912*inputs[604] -0.0255059*inputs[605] -0.0301618*inputs[606] -0.00216569*inputs[607] -0.0389482*inputs[608] -0.013151*inputs[609] +0.0071971*inputs[610] +0.020443*inputs[611] -0.00671582*inputs[612] -0.000456672*inputs[613] -0.00216568*inputs[614] -0.00216573*inputs[615] -0.0427879*inputs[616] +0.0158816*inputs[617] -0.0084509*inputs[618] -0.00433262*inputs[619] +0.00227569*inputs[620] -0.0012682*inputs[621] -0.0109847*inputs[622] +0.00280661*inputs[623] -0.0144188*inputs[624] +0.0118849*inputs[625] -0.00216568*inputs[626] -0.0124754*inputs[627] +0.0134062*inputs[628] +0.00704588*inputs[629] -0.051667*inputs[630] +0.0271996*inputs[631] -0.00294685*inputs[632] +0.0109976*inputs[633] +0.000644505*inputs[634] +0.00203356*inputs[635] +0.000644502*inputs[636] +0.0257884*inputs[637] +0.0223551*inputs[638] +0.012683*inputs[639] +0.0154272*inputs[640] -0.0171522*inputs[641] -0.00487147*inputs[642] -0.00216568*inputs[643] -0.00375525*inputs[644] -0.00826309*inputs[645] -0.0120823*inputs[646] +0.00131953*inputs[647] -0.00153047*inputs[648] -0.00380753*inputs[649] -0.00700033*inputs[650] -0.0121436*inputs[651] -0.0121436*inputs[652] -0.0121436*inputs[653] +0.00612895*inputs[654] -0.00153047*inputs[655] -0.00153047*inputs[656] -0.00153047*inputs[657] -0.00153047*inputs[658] -0.00153052*inputs[659] -0.00153048*inputs[660] -0.00153047*inputs[661] -0.00990002*inputs[662] -0.0108754*inputs[663] -0.0108754*inputs[664] -0.00153048*inputs[665] +0.011257*inputs[666] -0.0179266*inputs[667] -0.0179266*inputs[668] -0.0307788*inputs[669] +0.00849753*inputs[670] -0.015801*inputs[671] -0.00153047*inputs[672] +0.0048875*inputs[673] +0.00488751*inputs[674] +0.00488754*inputs[675] -0.0114232*inputs[676] -0.0114232*inputs[677] -0.0114232*inputs[678] -0.0226121*inputs[679] +0.0222716*inputs[680] +0.0222716*inputs[681] +0.0222716*inputs[682] -0.00153049*inputs[683] -0.00153049*inputs[684] -0.00153047*inputs[685] -0.00153051*inputs[686] -0.00153047*inputs[687] -0.0104403*inputs[688] -0.0104403*inputs[689] -0.00153047*inputs[690] -0.00153047*inputs[691] +0.00969346*inputs[692] +0.00969347*inputs[693] +0.00969343*inputs[694] +0.00969342*inputs[695] -0.00153047*inputs[696] -0.0015305*inputs[697] -0.00153047*inputs[698] +0.00169875*inputs[699] +0.00169874*inputs[700] -0.0125267*inputs[701] -0.00651491*inputs[702] -0.0065149*inputs[703] -0.00651488*inputs[704] -0.00651486*inputs[705] +0.00656602*inputs[706] -0.00153047*inputs[707] -0.00153047*inputs[708] -0.00153049*inputs[709] -0.00153047*inputs[710] -0.00153055*inputs[711] -0.00153052*inputs[712] -0.0593983*inputs[713] -0.0244384*inputs[714] -0.0418827*inputs[715] +0.0071631*inputs[716] +0.00716308*inputs[717] -0.00153047*inputs[718] -0.00153057*inputs[719] -0.0107654*inputs[720] -0.0107654*inputs[721] +0.000734549*inputs[722] -0.00261056*inputs[723] -0.00153049*inputs[724] -0.00153047*inputs[725] -0.00153048*inputs[726] -0.00153048*inputs[727] -0.00153047*inputs[728] -0.0330624*inputs[729] +0.00217244*inputs[730] +0.00217244*inputs[731] +0.00217244*inputs[732] +0.00217244*inputs[733] +0.00217244*inputs[734] -0.00153049*inputs[735] -0.00153047*inputs[736] -0.00153057*inputs[737] +0.0167813*inputs[738] -0.00153047*inputs[739] -0.00153049*inputs[740] -0.00153049*inputs[741] +0.0150008*inputs[742] +0.0150009*inputs[743] +0.00772584*inputs[744] +0.028756*inputs[745] +0.0243572*inputs[746] +0.0243572*inputs[747] -0.0120038*inputs[748] -0.0120038*inputs[749] +0.00649669*inputs[750] +0.00649668*inputs[751] +0.0117039*inputs[752] -0.0015305*inputs[753] -0.00153049*inputs[754] +0.00459153*inputs[755] +0.00459153*inputs[756] +0.00459153*inputs[757] +0.00459153*inputs[758] +0.0097958*inputs[759] +0.0097958*inputs[760] -0.00153047*inputs[761] -0.00204626*inputs[762] -0.00204625*inputs[763] -0.00204627*inputs[764] -0.00204624*inputs[765] -0.00204623*inputs[766] -0.00204623*inputs[767] -0.00829323*inputs[768] -0.00829322*inputs[769] -0.00153047*inputs[770] +0.012059*inputs[771] +0.012059*inputs[772] -0.00153047*inputs[773] -0.00153052*inputs[774] -0.00153048*inputs[775] -0.00153048*inputs[776] -0.00153047*inputs[777] -0.00153047*inputs[778] -0.00153047*inputs[779] -0.00153047*inputs[780] -0.0490621*inputs[781] -0.0245651*inputs[782] -0.0245651*inputs[783] -0.00846913*inputs[784] -0.00846914*inputs[785] -0.022057*inputs[786] -0.00153047*inputs[787] -0.00153047*inputs[788] -0.00153047*inputs[789] +0.0138556*inputs[790] -0.00153047*inputs[791] +0.00863541*inputs[792] -0.00153048*inputs[793] -0.00153047*inputs[794] +0.0095183*inputs[795] +0.0095183*inputs[796] +0.022993*inputs[797] -0.0015305*inputs[798] -0.00153047*inputs[799] -0.00153049*inputs[800] +0.00169576*inputs[801] +0.00169576*inputs[802] -0.0126879*inputs[803] -0.00153048*inputs[804] -0.00648178*inputs[805] -0.00153053*inputs[806] +0.00613629*inputs[807] -0.0015305*inputs[808] -0.00153047*inputs[809] -0.00153047*inputs[810] -0.00153056*inputs[811] -0.00648178*inputs[812] +0.034925*inputs[813] +0.00283009*inputs[814] +0.00283008*inputs[815] -0.00153047*inputs[816] +0.00613631*inputs[817] -0.00153047*inputs[818] -0.00153047*inputs[819] -0.00153047*inputs[820] -0.00153047*inputs[821] -0.0313227*inputs[822] -0.0267269*inputs[823] -0.00388937*inputs[824] -0.00153054*inputs[825] -0.0015305*inputs[826] +0.00613631*inputs[827] -0.00388937*inputs[828] +0.0309383*inputs[829] +0.0159952*inputs[830] +0.00673412*inputs[831] +0.00673415*inputs[832] +0.0115978*inputs[833] -0.00388938*inputs[834] +0.00673415*inputs[835] +0.00625255*inputs[836] +0.00767368*inputs[837] -0.00153055*inputs[838] +0.013518*inputs[839] -0.00153048*inputs[840] -0.000900244*inputs[841] +0.0179001*inputs[842] -0.00955317*inputs[843] -0.00955317*inputs[844] -0.00955317*inputs[845] -0.00955316*inputs[846] -0.00955316*inputs[847] +0.0235594*inputs[848] -0.01436*inputs[849] -0.01436*inputs[850] +0.0235594*inputs[851] -0.0237157*inputs[852] -0.0127387*inputs[853] +0.0142935*inputs[854] +0.00850701*inputs[855] -0.00153047*inputs[856] -0.00153047*inputs[857] -0.00153047*inputs[858] +0.00181138*inputs[859] -0.00153052*inputs[860] -0.00153056*inputs[861] -0.00153047*inputs[862] -0.00153047*inputs[863] -0.00153048*inputs[864] +0.0182502*inputs[865] -0.00342971*inputs[866] -0.0127387*inputs[867] -0.015832*inputs[868] -0.00153055*inputs[869] -0.00783255*inputs[870] -0.00783255*inputs[871] -0.00153047*inputs[872] -0.00740902*inputs[873] -0.00740903*inputs[874] -0.00740902*inputs[875] -0.00153052*inputs[876] +0.00625255*inputs[877] -0.022064*inputs[878] -0.0237589*inputs[879] -0.022064*inputs[880] +0.012008*inputs[881] -0.0361806*inputs[882] -0.00153055*inputs[883] -0.00153048*inputs[884] +0.00335138*inputs[885] +0.00335142*inputs[886] -0.00153047*inputs[887] -0.00153047*inputs[888] -0.00153057*inputs[889] -0.00153047*inputs[890] -0.00153052*inputs[891] -0.0015305*inputs[892] -0.00153047*inputs[893] -0.00153049*inputs[894] -0.00153047*inputs[895] -0.00153047*inputs[896] -0.00687682*inputs[897] -0.00153048*inputs[898] -0.00144451*inputs[899] -0.00687682*inputs[900] +0.00625255*inputs[901] +0.00799958*inputs[902] -0.0189449*inputs[903] -0.00153047*inputs[904] +0.00799958*inputs[905] +0.00799958*inputs[906] -0.000947739*inputs[907] -0.000947647*inputs[908] -0.00153047*inputs[909] -0.00153048*inputs[910] -0.00153047*inputs[911] +0.0270844*inputs[912] -0.0188502*inputs[913] -0.00153047*inputs[914] -0.00153053*inputs[915] -0.0189449*inputs[916] -0.00135105*inputs[917] -0.00153047*inputs[918] -0.00135103*inputs[919] -0.001351*inputs[920] -0.0189449*inputs[921] -0.037065*inputs[922] +0.0134636*inputs[923] -0.0125288*inputs[924] -0.00153051*inputs[925] -0.00153047*inputs[926] -0.00153052*inputs[927] -0.00153053*inputs[928] -0.0342882*inputs[929] -0.00153047*inputs[930] -0.00153048*inputs[931] -0.0305695*inputs[932] -0.0116823*inputs[933] +0.00893151*inputs[934] -0.00153047*inputs[935] -0.0116823*inputs[936] -0.006165*inputs[937] +0.0089315*inputs[938] -0.0157844*inputs[939] -0.00440754*inputs[940] -0.00440754*inputs[941] -0.00440754*inputs[942] -0.00440754*inputs[943] +0.000900198*inputs[944] -0.0179266*inputs[945] +0.000900221*inputs[946] +0.0242084*inputs[947] -0.00153049*inputs[948] +0.007819*inputs[949] +0.00781905*inputs[950] -0.0179266*inputs[951] -0.0198483*inputs[952] -0.00153048*inputs[953] -0.00153051*inputs[954] +0.00971465*inputs[955] +0.00971466*inputs[956] +0.0097147*inputs[957] +0.00971465*inputs[958] -0.0197791*inputs[959] -0.00153047*inputs[960] -0.00153053*inputs[961] +0.0104286*inputs[962] +0.0104286*inputs[963] +0.00452637*inputs[964] +0.0303304*inputs[965] -0.00153047*inputs[966] -0.00153047*inputs[967] -0.00153049*inputs[968] -0.0313232*inputs[969] -0.00153047*inputs[970] +0.0180157*inputs[971] -0.00153048*inputs[972] -0.00153051*inputs[973] -0.00153049*inputs[974] -0.00153051*inputs[975] -0.049679*inputs[976] -0.017956*inputs[977] +0.0129589*inputs[978] -0.00153048*inputs[979] -0.0207941*inputs[980] -0.00153049*inputs[981] +0.0115345*inputs[982] +0.0115346*inputs[983] -0.00153052*inputs[984] -0.00153047*inputs[985] -0.00153047*inputs[986] +0.00092634*inputs[987] -0.00153053*inputs[988] -0.00153047*inputs[989] -0.00153054*inputs[990] -0.00786244*inputs[991] -0.015834*inputs[992] -0.0015305*inputs[993] +0.00341161*inputs[994] -0.0015305*inputs[995] -0.0205403*inputs[996] +0.0311239*inputs[997] -0.00786246*inputs[998] -0.00153047*inputs[999] -0.00153047*inputs[1000] -0.00153047*inputs[1001] -0.0108082*inputs[1002] -0.00573454*inputs[1003] -0.00786243*inputs[1004] -0.00153047*inputs[1005] -0.00153051*inputs[1006] -0.00153047*inputs[1007] -0.0227766*inputs[1008] +0.0132148*inputs[1009] +0.0132148*inputs[1010] +0.0132148*inputs[1011] -0.00153047*inputs[1012] -0.00153047*inputs[1013] -0.00153047*inputs[1014] -0.00153048*inputs[1015] -0.00833407*inputs[1016] -0.00833407*inputs[1017] +0.0157545*inputs[1018] +0.0157545*inputs[1019] -0.0223229*inputs[1020] -0.000773687*inputs[1021] -0.00153047*inputs[1022] -0.00153047*inputs[1023] -0.00786243*inputs[1024] -0.00362463*inputs[1025] -0.00362463*inputs[1026] -0.00362463*inputs[1027] -0.00153047*inputs[1028] -0.00153047*inputs[1029] -0.00153047*inputs[1030] -0.00786245*inputs[1031] +0.00665584*inputs[1032] +0.0135598*inputs[1033] +0.00665583*inputs[1034] -0.0154863*inputs[1035] -0.0154863*inputs[1036] -0.0154863*inputs[1037] -0.0154863*inputs[1038] -0.0154863*inputs[1039] -0.00153047*inputs[1040] -0.00153047*inputs[1041] -0.00153047*inputs[1042] +0.0110368*inputs[1043] -0.0163445*inputs[1044] -0.0163444*inputs[1045] -0.0163445*inputs[1046] +0.0110368*inputs[1047] +0.0103424*inputs[1048] -0.00153047*inputs[1049] -0.0165725*inputs[1050] +0.0103424*inputs[1051] +0.0110369*inputs[1052] -0.00153047*inputs[1053] +0.000885273*inputs[1054] +0.000885272*inputs[1055] +0.000885272*inputs[1056] +0.000885267*inputs[1057] -0.0159624*inputs[1058] +0.0327767*inputs[1059] -0.00153049*inputs[1060] -0.00153048*inputs[1061] -0.00153056*inputs[1062] +0.0145062*inputs[1063] +0.0330156*inputs[1064] -0.00153047*inputs[1065] -0.0110552*inputs[1066] -0.0110552*inputs[1067] -0.00153049*inputs[1068] -0.00153047*inputs[1069] -0.00153047*inputs[1070] -0.00153052*inputs[1071] -0.00153047*inputs[1072] -0.00153047*inputs[1073] -0.00153053*inputs[1074] -0.00153047*inputs[1075] -0.00153048*inputs[1076] -0.00153049*inputs[1077] +0.00476636*inputs[1078] +0.0239796*inputs[1079] +0.0184421*inputs[1080] +0.00476638*inputs[1081] +0.00476636*inputs[1082] -0.00459349*inputs[1083] -0.00459352*inputs[1084] -0.00459352*inputs[1085] -0.00153047*inputs[1086] -0.00153048*inputs[1087] +0.00476638*inputs[1088] -0.0357302*inputs[1089] -0.00153047*inputs[1090] +0.0103424*inputs[1091] -0.0179804*inputs[1092] -0.00153047*inputs[1093] +0.0100954*inputs[1094] -0.00153047*inputs[1095] +0.0103424*inputs[1096] -0.00153048*inputs[1097] -0.00153051*inputs[1098] +0.000154318*inputs[1099] +0.000154334*inputs[1100] -0.00153052*inputs[1101] -0.00153047*inputs[1102] +0.0109675*inputs[1103] +0.0187757*inputs[1104] -0.00153047*inputs[1105] -0.00153047*inputs[1106] -0.00153048*inputs[1107] -0.00153047*inputs[1108] -0.0142954*inputs[1109] +0.00394979*inputs[1110] +0.0283359*inputs[1111] -0.0161034*inputs[1112] -0.00153047*inputs[1113] -0.0139634*inputs[1114] -0.00153047*inputs[1115] -0.0102422*inputs[1116] -0.0286757*inputs[1117] +0.0117849*inputs[1118] -0.00153051*inputs[1119] -0.00153049*inputs[1120] -0.00153051*inputs[1121] -0.00153049*inputs[1122] -0.00153047*inputs[1123] -0.00153047*inputs[1124] -0.00153051*inputs[1125] -0.00582054*inputs[1126] -0.00582054*inputs[1127] -0.00153048*inputs[1128] -0.0015305*inputs[1129] -0.0212943*inputs[1130] -0.0363037*inputs[1131] +0.024948*inputs[1132] -0.00843506*inputs[1133] -0.00843506*inputs[1134] -0.0103675*inputs[1135] -0.0103675*inputs[1136] -0.0103675*inputs[1137] -0.0161704*inputs[1138] -0.00153047*inputs[1139] -0.00153051*inputs[1140] +0.00603622*inputs[1141] -0.0139634*inputs[1142] -0.00263467*inputs[1143] -0.0026347*inputs[1144] -0.00263472*inputs[1145] -0.00263469*inputs[1146] +0.0202017*inputs[1147] +0.0202017*inputs[1148] +0.00850737*inputs[1149] -0.00153047*inputs[1150] -0.00153047*inputs[1151] -0.00153047*inputs[1152] +0.00654914*inputs[1153] +0.00654919*inputs[1154] +0.00654915*inputs[1155] -0.010563*inputs[1156] -0.0218273*inputs[1157] -0.0388732*inputs[1158] -0.00153048*inputs[1159] -0.00153047*inputs[1160] +0.00180436*inputs[1161] -0.00153047*inputs[1162] +0.0226333*inputs[1163] -0.00153048*inputs[1164] -0.00153051*inputs[1165] -0.00153047*inputs[1166] -0.0237536*inputs[1167] -0.00153047*inputs[1168] -0.00153048*inputs[1169] -0.0015305*inputs[1170] -0.00153047*inputs[1171] -0.00153048*inputs[1172] +0.0024418*inputs[1173] +0.00845211*inputs[1174] +0.0084521*inputs[1175] +0.00845212*inputs[1176] +0.00845208*inputs[1177] +0.011722*inputs[1178] +0.011722*inputs[1179] +0.011722*inputs[1180] +0.011722*inputs[1181] -0.00153047*inputs[1182] -0.00153047*inputs[1183] -0.00153047*inputs[1184] -0.00153051*inputs[1185] -0.0326712*inputs[1186] -0.00153047*inputs[1187] -0.00153047*inputs[1188] -0.00153049*inputs[1189] +0.0324855*inputs[1190] -0.00153048*inputs[1191] +0.00474746*inputs[1192] -0.0312206*inputs[1193] +0.0261035*inputs[1194] -0.00153051*inputs[1195] -0.00153047*inputs[1196] +0.0283718*inputs[1197] +0.0194582*inputs[1198] +0.0194582*inputs[1199] -0.02297*inputs[1200] -0.00153056*inputs[1201] -0.00153047*inputs[1202] -0.00153047*inputs[1203] -0.00153047*inputs[1204] -0.020217*inputs[1205] +0.00762757*inputs[1206] +0.00762757*inputs[1207] -0.00153048*inputs[1208] -0.00153047*inputs[1209] -0.00153047*inputs[1210] -0.0015305*inputs[1211] -0.00153055*inputs[1212] -0.00153048*inputs[1213] -0.00153047*inputs[1214] -0.00153047*inputs[1215] -0.00153047*inputs[1216] -0.00153047*inputs[1217] -0.00153049*inputs[1218] +0.0212419*inputs[1219] +0.0212419*inputs[1220] +0.0212419*inputs[1221] -0.012621*inputs[1222] -0.012621*inputs[1223] -0.012621*inputs[1224] -0.0186439*inputs[1225] -0.0186441*inputs[1226] -0.00153049*inputs[1227] -0.0015305*inputs[1228] -0.00153049*inputs[1229] -0.00555486*inputs[1230] -0.00555486*inputs[1231] -0.0313232*inputs[1232] -0.00153047*inputs[1233] -0.00153055*inputs[1234] -0.00153047*inputs[1235] +0.0305271*inputs[1236] -0.00153048*inputs[1237] -0.0147561*inputs[1238] -0.0168395*inputs[1239] -0.0168395*inputs[1240] -0.00153047*inputs[1241] -0.00153047*inputs[1242] -0.00153048*inputs[1243] -0.00153057*inputs[1244] -0.0015305*inputs[1245] -0.00716124*inputs[1246] -0.00716121*inputs[1247] -0.00716121*inputs[1248] +0.0133672*inputs[1249] -0.00153049*inputs[1250] -0.00153048*inputs[1251] -0.00781422*inputs[1252] -0.00153049*inputs[1253] -0.0170583*inputs[1254] -0.00543446*inputs[1255] -0.00543445*inputs[1256] -0.0165819*inputs[1257] -0.0101492*inputs[1258] -0.0101492*inputs[1259] +0.00339589*inputs[1260] +0.00425784*inputs[1261] +0.00339585*inputs[1262] +0.00339587*inputs[1263] +0.00339584*inputs[1264] -0.0317971*inputs[1265] +0.0356127*inputs[1266] -0.00153047*inputs[1267] -0.00153051*inputs[1268] -0.00153053*inputs[1269] +0.0212765*inputs[1270] +0.0183594*inputs[1271] -0.00735186*inputs[1272] -0.00735189*inputs[1273] +0.0097245*inputs[1274] +0.0097245*inputs[1275] -0.00153052*inputs[1276] -0.00153047*inputs[1277] +0.0212173*inputs[1278] -0.00153053*inputs[1279] -0.00153048*inputs[1280] +0.00425791*inputs[1281] -0.00153053*inputs[1282] -0.00153053*inputs[1283] -0.015242*inputs[1284] -0.0122235*inputs[1285] -0.0465061*inputs[1286] -0.0015305*inputs[1287] -0.00153049*inputs[1288] -0.00153048*inputs[1289] -0.00811342*inputs[1290] -0.0126268*inputs[1291] -0.0155478*inputs[1292] -0.0308029*inputs[1293] -0.0122241*inputs[1294] -0.0062262*inputs[1295] -0.0015305*inputs[1296] -0.0126268*inputs[1297] +0.0306367*inputs[1298] -0.00153047*inputs[1299] -0.00153047*inputs[1300] -0.00153047*inputs[1301] -0.00153048*inputs[1302] +0.00400572*inputs[1303] -0.00153049*inputs[1304] +0.0171925*inputs[1305] +0.0171925*inputs[1306] +0.0171924*inputs[1307] -0.00153053*inputs[1308] +0.00908989*inputs[1309] -0.00153047*inputs[1310] -0.00153047*inputs[1311] -0.00153047*inputs[1312] -0.0287839*inputs[1313] -0.0287839*inputs[1314] -0.00153048*inputs[1315] -0.00153047*inputs[1316] -0.00442899*inputs[1317] -0.00153048*inputs[1318] -0.0164551*inputs[1319] +0.0283931*inputs[1320] -0.0015305*inputs[1321] -0.00612798*inputs[1322] -0.00153049*inputs[1323] -0.00153047*inputs[1324] +0.0361507*inputs[1325] +0.00860433*inputs[1326] -0.00153047*inputs[1327] -0.00153049*inputs[1328] -0.00153048*inputs[1329] -0.00153053*inputs[1330] -0.00740937*inputs[1331] -0.00153051*inputs[1332] -0.00153047*inputs[1333] -0.00153048*inputs[1334] -0.00153047*inputs[1335] +0.0128975*inputs[1336] +0.0128975*inputs[1337] +0.0128976*inputs[1338] -0.010863*inputs[1339] -0.010863*inputs[1340] -0.00153047*inputs[1341] -0.00732708*inputs[1342] -0.00153047*inputs[1343] -0.00153047*inputs[1344] -0.00153047*inputs[1345] -0.0164861*inputs[1346] -0.00153047*inputs[1347] -0.00153047*inputs[1348] -0.00153047*inputs[1349] -0.00153047*inputs[1350] -0.00153047*inputs[1351] -0.00153048*inputs[1352] +0.0120212*inputs[1353] +0.0120212*inputs[1354] -0.00153047*inputs[1355] -0.00153047*inputs[1356] +0.0330153*inputs[1357] +0.0303139*inputs[1358] -0.00153048*inputs[1359] -0.00153052*inputs[1360] -0.00153048*inputs[1361] -0.00153047*inputs[1362] -0.00153048*inputs[1363] +0.021582*inputs[1364] +0.0215819*inputs[1365] +0.000968852*inputs[1366] +0.000968858*inputs[1367] +0.000968874*inputs[1368] +0.000968848*inputs[1369] +0.000968874*inputs[1370] -0.0232333*inputs[1371] -0.00620875*inputs[1372] -0.00620873*inputs[1373] +0.00814246*inputs[1374] -0.00326026*inputs[1375] -0.00326025*inputs[1376] -0.011288*inputs[1377] -0.0487327*inputs[1378] +0.00178474*inputs[1379] -0.00153047*inputs[1380] -0.00153047*inputs[1381] -0.00153047*inputs[1382] -0.00153051*inputs[1383] -0.0330408*inputs[1384] -0.0163341*inputs[1385] -0.0163341*inputs[1386] -0.00153048*inputs[1387] -0.00153047*inputs[1388] -0.00153047*inputs[1389] -0.0305561*inputs[1390] -0.0015305*inputs[1391] -0.00153047*inputs[1392] -0.00153047*inputs[1393] -0.00153049*inputs[1394] -0.00153053*inputs[1395] 
		combinations[4] = -0.0443664 -0.0845172*inputs[0] -0.0467854*inputs[1] -0.225244*inputs[2] -0.156326*inputs[3] -0.157091*inputs[4] -0.0824913*inputs[5] -0.0160329*inputs[6] -0.033285*inputs[7] -0.0509788*inputs[8] -0.0054863*inputs[9] -0.0117555*inputs[10] -0.0891715*inputs[11] -0.0354892*inputs[12] -0.10301*inputs[13] -0.183336*inputs[14] -0.034976*inputs[15] -0.00956596*inputs[16] -0.0200672*inputs[17] -0.112329*inputs[18] +0.10674*inputs[19] -0.00690463*inputs[20] -0.0168011*inputs[21] -0.0286915*inputs[22] -0.00429184*inputs[23] +0.00142304*inputs[24] +0.0201386*inputs[25] -0.129056*inputs[26] +0.0144412*inputs[27] -0.010097*inputs[28] -0.00132352*inputs[29] -0.0181317*inputs[30] +0.0208472*inputs[31] +0.0945627*inputs[32] -0.107046*inputs[33] +0.0804483*inputs[34] -0.107317*inputs[35] -0.0929327*inputs[36] -0.00856494*inputs[37] +0.00701296*inputs[38] +0.127568*inputs[39] -0.0414455*inputs[40] +0.187865*inputs[41] -0.0160291*inputs[42] +0.0450204*inputs[43] -0.0362241*inputs[44] -0.0130753*inputs[45] +0.0182895*inputs[46] +0.0645858*inputs[47] +0.080828*inputs[48] -0.0944484*inputs[49] -0.00304879*inputs[50] -0.00333998*inputs[51] -0.00262938*inputs[52] -0.014033*inputs[53] +0.00169793*inputs[54] +0.0584691*inputs[55] -0.063876*inputs[56] -0.0364733*inputs[57] +0.110542*inputs[58] +0.00280596*inputs[59] +0.131714*inputs[60] +0.0292658*inputs[61] +0.0295513*inputs[62] +0.0342345*inputs[63] -0.00174861*inputs[64] +0.15452*inputs[65] +0.082326*inputs[66] -0.047206*inputs[67] +0.0424263*inputs[68] -0.000253816*inputs[69] -0.0125529*inputs[70] +0.0026694*inputs[71] -0.0116313*inputs[72] +0.0121447*inputs[73] +0.0918813*inputs[74] -0.00903621*inputs[75] -0.103559*inputs[76] -0.0213546*inputs[77] -0.0190662*inputs[78] -0.0131169*inputs[79] +0.0303086*inputs[80] +0.0394111*inputs[81] -0.0115211*inputs[82] -0.0931377*inputs[83] -0.0227931*inputs[84] +0.043402*inputs[85] -0.044665*inputs[86] +0.0170581*inputs[87] +0.0252682*inputs[88] +0.0146823*inputs[89] +0.00491737*inputs[90] +0.0238415*inputs[91] -0.00157467*inputs[92] +0.0230798*inputs[93] +0.103719*inputs[94] +0.0686997*inputs[95] -0.00942623*inputs[96] -0.0252179*inputs[97] +0.0121536*inputs[98] -0.0374882*inputs[99] -0.00572547*inputs[100] +0.0296768*inputs[101] +0.0212715*inputs[102] +0.0128466*inputs[103] +0.047243*inputs[104] -0.00375494*inputs[105] +0.137443*inputs[106] -0.00451251*inputs[107] +0.0835894*inputs[108] -0.00650205*inputs[109] -0.0477854*inputs[110] -0.062743*inputs[111] +0.0194073*inputs[112] -0.0216269*inputs[113] +0.0629458*inputs[114] -0.0196823*inputs[115] +0.00263541*inputs[116] +0.0162678*inputs[117] +0.0323887*inputs[118] +0.0175434*inputs[119] +0.030411*inputs[120] -0.0833606*inputs[121] -0.0776129*inputs[122] +0.0809704*inputs[123] +0.0283631*inputs[124] +0.0150839*inputs[125] +0.00396415*inputs[126] +0.0031664*inputs[127] -0.00704496*inputs[128] -0.00477933*inputs[129] -0.00909833*inputs[130] -0.148187*inputs[131] -0.0218667*inputs[132] -0.0115797*inputs[133] +0.0201988*inputs[134] +0.0163812*inputs[135] +0.0829845*inputs[136] +0.0068595*inputs[137] +0.0216025*inputs[138] +0.0845554*inputs[139] -0.0108756*inputs[140] -0.0217471*inputs[141] -0.0293842*inputs[142] -0.0418121*inputs[143] +0.013229*inputs[144] -0.00722938*inputs[145] -0.0826009*inputs[146] +0.0212229*inputs[147] +0.0250088*inputs[148] +0.0377235*inputs[149] +0.020348*inputs[150] +0.0141971*inputs[151] +0.0326334*inputs[152] +0.000545369*inputs[153] +0.0622096*inputs[154] +0.0167544*inputs[155] -0.00264924*inputs[156] +0.00926362*inputs[157] +0.0255363*inputs[158] -0.0647415*inputs[159] +0.00318406*inputs[160] +0.0140418*inputs[161] -0.00425547*inputs[162] +0.0233277*inputs[163] +0.00777625*inputs[164] +0.00153055*inputs[165] -0.0212889*inputs[166] -0.00744501*inputs[167] +0.000326188*inputs[168] +0.0234481*inputs[169] +0.0125243*inputs[170] +0.0372075*inputs[171] -0.00679127*inputs[172] -0.0227612*inputs[173] +0.0183773*inputs[174] -0.0598695*inputs[175] -0.0121262*inputs[176] -0.00165267*inputs[177] +0.0703425*inputs[178] -0.0471027*inputs[179] +0.0569143*inputs[180] -0.00217693*inputs[181] -0.00302749*inputs[182] -0.0110552*inputs[183] +0.0111069*inputs[184] +0.00136068*inputs[185] +0.0117756*inputs[186] +0.0119203*inputs[187] -0.0159558*inputs[188] +0.00865395*inputs[189] +0.0255299*inputs[190] -0.0076668*inputs[191] -0.0358751*inputs[192] +0.00335326*inputs[193] -0.0492847*inputs[194] -0.0198295*inputs[195] -0.0205805*inputs[196] +0.0368091*inputs[197] -0.0349151*inputs[198] +0.00605037*inputs[199] +0.00339846*inputs[200] -0.00455523*inputs[201] +0.00382931*inputs[202] +0.0129844*inputs[203] -0.0278904*inputs[204] +0.040539*inputs[205] -0.0230726*inputs[206] +0.0268422*inputs[207] +0.0175985*inputs[208] +0.0537925*inputs[209] -0.00109465*inputs[210] +0.00519612*inputs[211] +0.027844*inputs[212] +0.00320518*inputs[213] -0.0163962*inputs[214] -0.0555447*inputs[215] -0.02749*inputs[216] +0.0218582*inputs[217] -0.014989*inputs[218] +0.00729455*inputs[219] -0.016963*inputs[220] +0.0188308*inputs[221] -0.0140601*inputs[222] -0.00290957*inputs[223] +0.0389663*inputs[224] +0.0536009*inputs[225] -0.000712544*inputs[226] +0.00732848*inputs[227] -0.00934779*inputs[228] -0.0161766*inputs[229] -0.0199277*inputs[230] -0.00491081*inputs[231] -0.0196686*inputs[232] -0.0101421*inputs[233] -0.00180279*inputs[234] -0.0392134*inputs[235] +0.00107122*inputs[236] -0.0178491*inputs[237] +0.0111515*inputs[238] +0.0414664*inputs[239] +0.00314287*inputs[240] +0.00397277*inputs[241] +0.00519766*inputs[242] +0.0194172*inputs[243] +0.012905*inputs[244] -0.0376073*inputs[245] +0.0156085*inputs[246] +0.00404177*inputs[247] +0.0472774*inputs[248] +0.00776053*inputs[249] -0.0100016*inputs[250] +0.0279114*inputs[251] +0.0125378*inputs[252] +0.0365989*inputs[253] +0.0135418*inputs[254] +0.00722412*inputs[255] -0.0127978*inputs[256] +0.00229074*inputs[257] -0.00835147*inputs[258] +0.00636265*inputs[259] +0.00280964*inputs[260] -0.030309*inputs[261] -0.0156037*inputs[262] +0.00124577*inputs[263] -0.00327819*inputs[264] +0.0089218*inputs[265] -0.026088*inputs[266] +0.0594251*inputs[267] +0.0220337*inputs[268] -0.0108854*inputs[269] +0.00855625*inputs[270] +0.0536034*inputs[271] +0.00453424*inputs[272] -0.00714748*inputs[273] +0.0028096*inputs[274] +0.00280961*inputs[275] -0.0102839*inputs[276] +0.00344846*inputs[277] +0.0500502*inputs[278] -0.00140468*inputs[279] -0.0232915*inputs[280] +0.000178594*inputs[281] -0.0184068*inputs[282] -0.0305008*inputs[283] -0.0212901*inputs[284] -0.0818375*inputs[285] +0.00828724*inputs[286] +0.0257584*inputs[287] +0.0165167*inputs[288] -0.0219672*inputs[289] -0.0265581*inputs[290] +0.00378039*inputs[291] +0.00840653*inputs[292] +0.0166812*inputs[293] -0.0196324*inputs[294] -0.00404864*inputs[295] +0.0177726*inputs[296] -0.00159163*inputs[297] -0.0214418*inputs[298] -0.0493368*inputs[299] -0.0336713*inputs[300] -0.00945454*inputs[301] -0.00553673*inputs[302] +0.088157*inputs[303] -0.00594711*inputs[304] +0.0430108*inputs[305] +0.0177641*inputs[306] +0.0394341*inputs[307] +0.0195107*inputs[308] +0.0202321*inputs[309] +0.00229075*inputs[310] +0.00863412*inputs[311] +0.0553911*inputs[312] -0.00975854*inputs[313] -0.0410102*inputs[314] -0.010127*inputs[315] +0.00624219*inputs[316] -0.0124065*inputs[317] -0.00139325*inputs[318] -0.029275*inputs[319] +0.0279523*inputs[320] +0.00696686*inputs[321] +0.00528852*inputs[322] +0.0174051*inputs[323] +0.00367649*inputs[324] +0.000664149*inputs[325] +0.00600244*inputs[326] -0.0046531*inputs[327] +0.000988727*inputs[328] +0.0024319*inputs[329] +0.012743*inputs[330] -0.0175163*inputs[331] -0.0293882*inputs[332] +0.00528851*inputs[333] +0.0252557*inputs[334] +0.0227335*inputs[335] +0.00812819*inputs[336] -0.0160343*inputs[337] +0.041455*inputs[338] +0.00623355*inputs[339] +0.00472291*inputs[340] -0.0496339*inputs[341] +0.000159458*inputs[342] -0.0108547*inputs[343] +0.0231309*inputs[344] -0.0304537*inputs[345] +0.0220004*inputs[346] +0.00188251*inputs[347] +0.0200538*inputs[348] -0.0146131*inputs[349] +8.75188e-05*inputs[350] -0.0403499*inputs[351] +0.0060423*inputs[352] -0.010479*inputs[353] +0.013517*inputs[354] -0.0092561*inputs[355] -0.00455652*inputs[356] -0.0117646*inputs[357] +0.0024319*inputs[358] -0.00661882*inputs[359] -0.0139435*inputs[360] -0.000320186*inputs[361] +0.00258466*inputs[362] +0.0126146*inputs[363] -0.0123773*inputs[364] +0.0174014*inputs[365] -0.000111822*inputs[366] +0.0228884*inputs[367] -0.0208219*inputs[368] +0.0129044*inputs[369] +0.0024319*inputs[370] +0.0024319*inputs[371] +0.00910708*inputs[372] +0.0119594*inputs[373] -0.0198434*inputs[374] +0.00976484*inputs[375] -0.0102682*inputs[376] -0.0354439*inputs[377] -0.0124163*inputs[378] -0.0181114*inputs[379] -0.0030263*inputs[380] -0.0430225*inputs[381] +0.0253378*inputs[382] +0.0119123*inputs[383] +0.00132611*inputs[384] +0.00556197*inputs[385] +0.00399684*inputs[386] -0.00738258*inputs[387] +0.0024319*inputs[388] +0.013877*inputs[389] +0.0082317*inputs[390] -0.00544171*inputs[391] +0.0024319*inputs[392] -0.0314716*inputs[393] -0.0122263*inputs[394] +0.0411569*inputs[395] -0.0043359*inputs[396] -0.0109159*inputs[397] -0.0135819*inputs[398] +0.0143316*inputs[399] -0.0296538*inputs[400] -0.0260735*inputs[401] +0.010877*inputs[402] -0.0230323*inputs[403] +0.00653255*inputs[404] -0.000337765*inputs[405] +0.0638399*inputs[406] -0.000413493*inputs[407] +0.0279328*inputs[408] +0.0125701*inputs[409] -0.000452216*inputs[410] +0.0459191*inputs[411] -0.00431049*inputs[412] +0.0024319*inputs[413] +0.0196947*inputs[414] -0.00334004*inputs[415] +0.0239577*inputs[416] -0.000862719*inputs[417] +0.0168659*inputs[418] +0.0114264*inputs[419] +0.000427002*inputs[420] +0.0305399*inputs[421] +0.00243192*inputs[422] +0.0255142*inputs[423] +0.0142154*inputs[424] +0.0183545*inputs[425] +0.0114686*inputs[426] +0.0372674*inputs[427] -0.0126604*inputs[428] -0.00890535*inputs[429] -0.00940368*inputs[430] -0.0120733*inputs[431] +0.048285*inputs[432] -0.0114826*inputs[433] +0.0184517*inputs[434] +0.00899874*inputs[435] +0.00533773*inputs[436] -0.00395763*inputs[437] -0.00103798*inputs[438] +0.0335287*inputs[439] +0.00378081*inputs[440] +0.00243193*inputs[441] +0.00790786*inputs[442] +0.0160472*inputs[443] +0.00715187*inputs[444] +0.0335296*inputs[445] +0.0756011*inputs[446] -0.0267699*inputs[447] +0.0180874*inputs[448] +0.0118344*inputs[449] +0.00198458*inputs[450] +0.0390523*inputs[451] +0.0120804*inputs[452] +0.00927199*inputs[453] -0.0161438*inputs[454] +0.0247493*inputs[455] +0.0247493*inputs[456] -0.00500825*inputs[457] -0.00443639*inputs[458] -0.054917*inputs[459] +0.00198457*inputs[460] +0.0304393*inputs[461] +0.0235625*inputs[462] +0.00140247*inputs[463] +0.0252953*inputs[464] +0.00198459*inputs[465] -0.0173638*inputs[466] +0.0668416*inputs[467] -0.00779514*inputs[468] -0.00790928*inputs[469] -0.00178922*inputs[470] -0.0256128*inputs[471] -0.0342486*inputs[472] +0.0134422*inputs[473] -0.00782331*inputs[474] -0.0116358*inputs[475] +0.0142461*inputs[476] +0.0101498*inputs[477] +0.0807413*inputs[478] +0.0268364*inputs[479] +0.0229083*inputs[480] -0.0239785*inputs[481] +0.00365695*inputs[482] +0.0120589*inputs[483] +0.00198456*inputs[484] +0.00308065*inputs[485] +0.00906016*inputs[486] +0.00198456*inputs[487] -0.0266474*inputs[488] +0.0123422*inputs[489] +0.0936658*inputs[490] +0.00197533*inputs[491] -0.00731787*inputs[492] +0.0222652*inputs[493] -0.00278042*inputs[494] -0.00534177*inputs[495] +0.0264211*inputs[496] +0.0298112*inputs[497] +0.0180646*inputs[498] -0.00420592*inputs[499] -0.0417399*inputs[500] -0.0156631*inputs[501] +0.0191265*inputs[502] +0.00198456*inputs[503] +0.0309489*inputs[504] +0.00198458*inputs[505] -0.00657021*inputs[506] -0.0298276*inputs[507] -0.00083396*inputs[508] +0.0248788*inputs[509] -0.0274356*inputs[510] -0.0130042*inputs[511] +0.039506*inputs[512] +0.0156763*inputs[513] +0.00198457*inputs[514] +0.0406698*inputs[515] +0.0177172*inputs[516] +0.00198456*inputs[517] +0.00268895*inputs[518] +0.0115818*inputs[519] +0.0205776*inputs[520] +0.0101559*inputs[521] +0.00985027*inputs[522] +0.00985024*inputs[523] +0.0405467*inputs[524] -0.0216143*inputs[525] +0.0131192*inputs[526] +0.0110717*inputs[527] -0.00340766*inputs[528] +0.00198456*inputs[529] +0.0043261*inputs[530] -0.008629*inputs[531] -0.00862898*inputs[532] +0.00198456*inputs[533] -0.0143548*inputs[534] -0.00909471*inputs[535] +0.0131211*inputs[536] +0.00563646*inputs[537] +0.00908428*inputs[538] -0.014728*inputs[539] -0.014728*inputs[540] +0.00224621*inputs[541] +0.00258146*inputs[542] -0.00166465*inputs[543] +0.0390014*inputs[544] +0.00198456*inputs[545] +0.00198459*inputs[546] -0.00544015*inputs[547] +0.0127119*inputs[548] +0.00198459*inputs[549] +0.00198456*inputs[550] +0.0330236*inputs[551] +0.0211048*inputs[552] -0.00909473*inputs[553] +0.00620941*inputs[554] +0.00198459*inputs[555] -0.00388628*inputs[556] +0.0334437*inputs[557] +0.00327658*inputs[558] +0.0458429*inputs[559] +0.0037232*inputs[560] -0.0236417*inputs[561] +0.000468641*inputs[562] +0.0399059*inputs[563] -0.040163*inputs[564] -0.0137349*inputs[565] +0.00413354*inputs[566] +0.0231167*inputs[567] +0.00198456*inputs[568] -0.00985399*inputs[569] -0.00594052*inputs[570] +0.0273918*inputs[571] +0.0131246*inputs[572] -0.00447773*inputs[573] -0.0096858*inputs[574] +0.0152203*inputs[575] -0.00293795*inputs[576] +0.00197534*inputs[577] +0.00198456*inputs[578] +0.0230782*inputs[579] -0.0192146*inputs[580] +0.00140251*inputs[581] +0.00738906*inputs[582] +0.0144494*inputs[583] +0.000838462*inputs[584] +0.0278801*inputs[585] -0.00469693*inputs[586] -0.0144957*inputs[587] +0.00198456*inputs[588] +0.00198456*inputs[589] +0.0248517*inputs[590] -0.000833924*inputs[591] +0.032846*inputs[592] +0.00524781*inputs[593] +0.0103754*inputs[594] -0.00772149*inputs[595] -0.021988*inputs[596] -0.00161472*inputs[597] +0.00436576*inputs[598] -0.00984497*inputs[599] -0.00784226*inputs[600] +0.00159396*inputs[601] +0.00198456*inputs[602] -0.0257747*inputs[603] -0.0215946*inputs[604] +0.025779*inputs[605] +0.0299192*inputs[606] +0.00198456*inputs[607] +0.038918*inputs[608] +0.0138308*inputs[609] -0.00804689*inputs[610] -0.0159929*inputs[611] +0.00639452*inputs[612] +0.00064357*inputs[613] +0.00198457*inputs[614] +0.00198457*inputs[615] +0.0466762*inputs[616] -0.0167612*inputs[617] +0.00836443*inputs[618] +0.00413351*inputs[619] -0.0015142*inputs[620] +0.00129412*inputs[621] +0.011383*inputs[622] -0.00287693*inputs[623] +0.0161174*inputs[624] -0.0126914*inputs[625] +0.00198458*inputs[626] +0.0123129*inputs[627] -0.0144957*inputs[628] -0.0100567*inputs[629] +0.0533376*inputs[630] -0.0287989*inputs[631] +0.00268892*inputs[632] -0.0157761*inputs[633] -0.000179418*inputs[634] -0.00154595*inputs[635] -0.000179446*inputs[636] -0.026607*inputs[637] -0.0235631*inputs[638] -0.0129964*inputs[639] -0.0168685*inputs[640] +0.0178036*inputs[641] +0.00483559*inputs[642] +0.00198456*inputs[643] +0.00350987*inputs[644] +0.00848043*inputs[645] +0.0120803*inputs[646] -0.00149856*inputs[647] +0.00140251*inputs[648] +0.00352294*inputs[649] +0.00598444*inputs[650] +0.0117033*inputs[651] +0.0117034*inputs[652] +0.0117033*inputs[653] -0.00668798*inputs[654] +0.00140246*inputs[655] +0.00140246*inputs[656] +0.00140249*inputs[657] +0.00140249*inputs[658] +0.00140246*inputs[659] +0.00140247*inputs[660] +0.00140247*inputs[661] +0.00825509*inputs[662] +0.0109123*inputs[663] +0.0109123*inputs[664] +0.00140254*inputs[665] -0.0105655*inputs[666] +0.0178952*inputs[667] +0.0178952*inputs[668] +0.0313429*inputs[669] -0.00871621*inputs[670] +0.0152319*inputs[671] +0.00140247*inputs[672] -0.00506928*inputs[673] -0.00506927*inputs[674] -0.00506928*inputs[675] +0.0114039*inputs[676] +0.0114039*inputs[677] +0.0114039*inputs[678] +0.0227783*inputs[679] -0.0220393*inputs[680] -0.0220392*inputs[681] -0.0220392*inputs[682] +0.00140248*inputs[683] +0.00140251*inputs[684] +0.00140246*inputs[685] +0.0014025*inputs[686] +0.00140246*inputs[687] +0.00963557*inputs[688] +0.00963562*inputs[689] +0.00140249*inputs[690] +0.00140248*inputs[691] -0.0106898*inputs[692] -0.0106898*inputs[693] -0.0106898*inputs[694] -0.0106898*inputs[695] +0.00140246*inputs[696] +0.00140252*inputs[697] +0.00140246*inputs[698] -0.00215705*inputs[699] -0.00215701*inputs[700] +0.0125208*inputs[701] +0.00634543*inputs[702] +0.00634538*inputs[703] +0.00634539*inputs[704] +0.00634539*inputs[705] -0.00698997*inputs[706] +0.00140251*inputs[707] +0.00140247*inputs[708] +0.00140246*inputs[709] +0.00140247*inputs[710] +0.00140247*inputs[711] +0.00140247*inputs[712] +0.0633075*inputs[713] +0.0238442*inputs[714] +0.0432798*inputs[715] -0.00763681*inputs[716] -0.00763681*inputs[717] +0.00140246*inputs[718] +0.00140247*inputs[719] +0.0105491*inputs[720] +0.0105491*inputs[721] -0.000740222*inputs[722] +0.00248761*inputs[723] +0.00140246*inputs[724] +0.00140246*inputs[725] +0.00140251*inputs[726] +0.00140251*inputs[727] +0.00140249*inputs[728] +0.0368754*inputs[729] -0.00293795*inputs[730] -0.00293795*inputs[731] -0.00293795*inputs[732] -0.00293795*inputs[733] -0.00293795*inputs[734] +0.00140247*inputs[735] +0.00140248*inputs[736] +0.00140252*inputs[737] -0.0176877*inputs[738] +0.00140246*inputs[739] +0.00140249*inputs[740] +0.0014025*inputs[741] -0.0154698*inputs[742] -0.0154698*inputs[743] -0.00596955*inputs[744] -0.0300569*inputs[745] -0.0277636*inputs[746] -0.0277636*inputs[747] +0.0133468*inputs[748] +0.0133468*inputs[749] -0.00679042*inputs[750] -0.00679042*inputs[751] -0.0127771*inputs[752] +0.00140247*inputs[753] +0.0014025*inputs[754] -0.00460142*inputs[755] -0.00460142*inputs[756] -0.00460141*inputs[757] -0.00460141*inputs[758] -0.00993331*inputs[759] -0.00993335*inputs[760] +0.00140246*inputs[761] +0.00166689*inputs[762] +0.00166695*inputs[763] +0.0016669*inputs[764] +0.0016669*inputs[765] +0.00166688*inputs[766] +0.00166688*inputs[767] +0.0107704*inputs[768] +0.0107704*inputs[769] +0.00140248*inputs[770] -0.0121506*inputs[771] -0.0121506*inputs[772] +0.00140246*inputs[773] +0.00140251*inputs[774] +0.00140247*inputs[775] +0.00140249*inputs[776] +0.00140247*inputs[777] +0.00140247*inputs[778] +0.00140247*inputs[779] +0.00140247*inputs[780] +0.0530578*inputs[781] +0.0236757*inputs[782] +0.0236757*inputs[783] +0.00820694*inputs[784] +0.00820694*inputs[785] +0.020608*inputs[786] +0.00140248*inputs[787] +0.00140251*inputs[788] +0.00140246*inputs[789] -0.0146985*inputs[790] +0.00140249*inputs[791] +0.000280847*inputs[792] +0.00140252*inputs[793] +0.00140248*inputs[794] -0.00913414*inputs[795] -0.00913414*inputs[796] -0.0235883*inputs[797] +0.00140247*inputs[798] +0.00140249*inputs[799] +0.0014025*inputs[800] -0.00183586*inputs[801] -0.00183585*inputs[802] +0.0124003*inputs[803] +0.00140246*inputs[804] +0.00641208*inputs[805] +0.00140246*inputs[806] -0.00612586*inputs[807] +0.00140249*inputs[808] +0.00140251*inputs[809] +0.00140249*inputs[810] +0.00140246*inputs[811] +0.00641208*inputs[812] -0.0391774*inputs[813] -0.00286095*inputs[814] -0.00286097*inputs[815] +0.00140246*inputs[816] -0.00612592*inputs[817] +0.00140247*inputs[818] +0.00140249*inputs[819] +0.00140246*inputs[820] +0.00140246*inputs[821] +0.0313484*inputs[822] +0.0266812*inputs[823] +0.00452538*inputs[824] +0.00140248*inputs[825] +0.00140246*inputs[826] -0.00612592*inputs[827] +0.0045254*inputs[828] -0.0327865*inputs[829] -0.0142581*inputs[830] -0.00734773*inputs[831] -0.00734775*inputs[832] -0.0126516*inputs[833] +0.00452537*inputs[834] -0.00734774*inputs[835] -0.00626367*inputs[836] -0.00689606*inputs[837] +0.00140249*inputs[838] -0.014289*inputs[839] +0.00140248*inputs[840] +0.000807473*inputs[841] -0.0184925*inputs[842] +0.00976774*inputs[843] +0.00976772*inputs[844] +0.00976772*inputs[845] +0.00976773*inputs[846] +0.00976772*inputs[847] -0.024712*inputs[848] +0.0148182*inputs[849] +0.0148181*inputs[850] -0.024712*inputs[851] +0.0234371*inputs[852] +0.0129527*inputs[853] -0.0139659*inputs[854] -0.00892954*inputs[855] +0.00140253*inputs[856] +0.00140254*inputs[857] +0.00140251*inputs[858] -0.00206666*inputs[859] +0.00140246*inputs[860] +0.00140246*inputs[861] +0.00140246*inputs[862] +0.00140246*inputs[863] +0.00140246*inputs[864] -0.0178048*inputs[865] +0.00322877*inputs[866] +0.0129527*inputs[867] +0.0155121*inputs[868] +0.00140247*inputs[869] +0.00800608*inputs[870] +0.00800609*inputs[871] +0.00140252*inputs[872] +0.00731419*inputs[873] +0.00731423*inputs[874] +0.00731419*inputs[875] +0.00140252*inputs[876] -0.00626367*inputs[877] +0.0227099*inputs[878] +0.0237478*inputs[879] +0.0227099*inputs[880] -0.0105435*inputs[881] +0.0365625*inputs[882] +0.00140246*inputs[883] +0.00140248*inputs[884] -0.0034021*inputs[885] -0.00340209*inputs[886] +0.00140246*inputs[887] +0.00140246*inputs[888] +0.00140248*inputs[889] +0.00140246*inputs[890] +0.00140255*inputs[891] +0.00140249*inputs[892] +0.00140251*inputs[893] +0.00140255*inputs[894] +0.00140246*inputs[895] +0.00140247*inputs[896] +0.00666087*inputs[897] +0.00140246*inputs[898] +0.00132405*inputs[899] +0.00666085*inputs[900] -0.00626367*inputs[901] -0.0082624*inputs[902] +0.0189855*inputs[903] +0.00140251*inputs[904] -0.00826233*inputs[905] -0.00826241*inputs[906] +0.00087342*inputs[907] +0.000873444*inputs[908] +0.00140251*inputs[909] +0.00140246*inputs[910] +0.00140248*inputs[911] -0.0294662*inputs[912] +0.0213794*inputs[913] +0.00140246*inputs[914] +0.00140253*inputs[915] +0.0189855*inputs[916] +0.00123131*inputs[917] +0.00140246*inputs[918] +0.00123131*inputs[919] +0.00123136*inputs[920] +0.0189855*inputs[921] +0.0380061*inputs[922] -0.0143075*inputs[923] +0.0123117*inputs[924] +0.00140247*inputs[925] +0.00140253*inputs[926] +0.00140253*inputs[927] +0.00140246*inputs[928] +0.032229*inputs[929] +0.00140246*inputs[930] +0.00140253*inputs[931] +0.0313105*inputs[932] +0.0120589*inputs[933] -0.00755527*inputs[934] +0.00140248*inputs[935] +0.0120589*inputs[936] +0.00804621*inputs[937] -0.00755527*inputs[938] +0.0156981*inputs[939] +0.00376647*inputs[940] +0.00376647*inputs[941] +0.00376648*inputs[942] +0.00376648*inputs[943] -0.00063705*inputs[944] +0.0178954*inputs[945] -0.000637057*inputs[946] -0.024222*inputs[947] +0.00140254*inputs[948] -0.0080418*inputs[949] -0.00804183*inputs[950] +0.0178954*inputs[951] +0.0219785*inputs[952] +0.00140247*inputs[953] +0.00140246*inputs[954] -0.010264*inputs[955] -0.010264*inputs[956] -0.0102639*inputs[957] -0.010264*inputs[958] +0.0203602*inputs[959] +0.00140247*inputs[960] +0.00140252*inputs[961] -0.0105529*inputs[962] -0.0105529*inputs[963] -0.00406404*inputs[964] -0.0298921*inputs[965] +0.00140246*inputs[966] +0.00140247*inputs[967] +0.00140246*inputs[968] +0.0313483*inputs[969] +0.00140246*inputs[970] -0.0188215*inputs[971] +0.00140251*inputs[972] +0.00140247*inputs[973] +0.00140254*inputs[974] +0.00140248*inputs[975] +0.0519177*inputs[976] +0.0191262*inputs[977] -0.0142131*inputs[978] +0.00140246*inputs[979] +0.0217917*inputs[980] +0.00140246*inputs[981] -0.024432*inputs[982] -0.024432*inputs[983] +0.00140246*inputs[984] +0.00140246*inputs[985] +0.00140247*inputs[986] -0.000527569*inputs[987] +0.00140246*inputs[988] +0.00140246*inputs[989] +0.00140253*inputs[990] +0.00798064*inputs[991] +0.0158667*inputs[992] +0.00140246*inputs[993] -0.0030928*inputs[994] +0.00140247*inputs[995] +0.0199321*inputs[996] -0.0310589*inputs[997] +0.00798065*inputs[998] +0.00140246*inputs[999] +0.00140253*inputs[1000] +0.00140251*inputs[1001] +0.0107204*inputs[1002] +0.00476842*inputs[1003] +0.00798066*inputs[1004] +0.00140252*inputs[1005] +0.00140247*inputs[1006] +0.00140247*inputs[1007] +0.0227973*inputs[1008] -0.012317*inputs[1009] -0.012317*inputs[1010] -0.012317*inputs[1011] +0.00140248*inputs[1012] +0.00140246*inputs[1013] +0.00140246*inputs[1014] +0.00140247*inputs[1015] +0.00931533*inputs[1016] +0.00931533*inputs[1017] -0.0153187*inputs[1018] -0.0153187*inputs[1019] +0.0241317*inputs[1020] +0.000850409*inputs[1021] +0.00140246*inputs[1022] +0.00140246*inputs[1023] +0.00798064*inputs[1024] +0.00358014*inputs[1025] +0.00358013*inputs[1026] +0.00358011*inputs[1027] +0.00140246*inputs[1028] +0.00140247*inputs[1029] +0.00140246*inputs[1030] +0.00798063*inputs[1031] -0.00621944*inputs[1032] -0.0144758*inputs[1033] -0.00621943*inputs[1034] +0.0160793*inputs[1035] +0.0160793*inputs[1036] +0.0160793*inputs[1037] +0.0160793*inputs[1038] +0.0160793*inputs[1039] +0.00140249*inputs[1040] +0.00140246*inputs[1041] +0.00140246*inputs[1042] -0.0114284*inputs[1043] +0.0166687*inputs[1044] +0.0166687*inputs[1045] +0.0166687*inputs[1046] -0.0114284*inputs[1047] -0.0108118*inputs[1048] +0.00140249*inputs[1049] +0.0173547*inputs[1050] -0.0108118*inputs[1051] -0.0114284*inputs[1052] +0.00140246*inputs[1053] -0.000492985*inputs[1054] -0.000492955*inputs[1055] -0.000493017*inputs[1056] -0.000492978*inputs[1057] +0.0149683*inputs[1058] -0.0326172*inputs[1059] +0.00140252*inputs[1060] +0.00140246*inputs[1061] +0.00140246*inputs[1062] -0.0153313*inputs[1063] -0.0346658*inputs[1064] +0.00140249*inputs[1065] +0.0114379*inputs[1066] +0.0114379*inputs[1067] +0.00140251*inputs[1068] +0.0014025*inputs[1069] +0.00140247*inputs[1070] +0.00140251*inputs[1071] +0.00140248*inputs[1072] +0.00140249*inputs[1073] +0.00140246*inputs[1074] +0.00140246*inputs[1075] +0.00140246*inputs[1076] +0.00140246*inputs[1077] -0.00507078*inputs[1078] -0.0250948*inputs[1079] -0.0184819*inputs[1080] -0.00507073*inputs[1081] -0.0050708*inputs[1082] +0.00444012*inputs[1083] +0.00444008*inputs[1084] +0.00444005*inputs[1085] +0.00140246*inputs[1086] +0.00140246*inputs[1087] -0.00507068*inputs[1088] +0.0387786*inputs[1089] +0.00140246*inputs[1090] -0.0108118*inputs[1091] +0.0201244*inputs[1092] +0.00140247*inputs[1093] -0.0104037*inputs[1094] +0.00140246*inputs[1095] -0.0108118*inputs[1096] +0.00140251*inputs[1097] +0.00140252*inputs[1098] +0.000387661*inputs[1099] +0.00038762*inputs[1100] +0.00140247*inputs[1101] +0.00140251*inputs[1102] -0.0107335*inputs[1103] -0.01942*inputs[1104] +0.00140246*inputs[1105] +0.00140255*inputs[1106] +0.00140249*inputs[1107] +0.00140246*inputs[1108] +0.0140909*inputs[1109] -0.00375573*inputs[1110] -0.0304933*inputs[1111] +0.0160016*inputs[1112] +0.00140246*inputs[1113] +0.0147813*inputs[1114] +0.00140251*inputs[1115] +0.00626718*inputs[1116] +0.0288197*inputs[1117] -0.00991884*inputs[1118] +0.00140246*inputs[1119] +0.00140251*inputs[1120] +0.00140247*inputs[1121] +0.00140248*inputs[1122] +0.00140247*inputs[1123] +0.00140246*inputs[1124] +0.00140247*inputs[1125] +0.00552895*inputs[1126] +0.00552897*inputs[1127] +0.00140247*inputs[1128] +0.00140247*inputs[1129] +0.0239901*inputs[1130] +0.0373158*inputs[1131] -0.0258395*inputs[1132] +0.00833466*inputs[1133] +0.00833466*inputs[1134] +0.010678*inputs[1135] +0.010678*inputs[1136] +0.010678*inputs[1137] +0.0160488*inputs[1138] +0.00140248*inputs[1139] +0.00140249*inputs[1140] -0.0053329*inputs[1141] +0.0147813*inputs[1142] +0.00239806*inputs[1143] +0.00239806*inputs[1144] +0.00239807*inputs[1145] +0.00239805*inputs[1146] -0.0204825*inputs[1147] -0.0204825*inputs[1148] -0.00856574*inputs[1149] +0.00140246*inputs[1150] +0.00140246*inputs[1151] +0.00140249*inputs[1152] -0.00698255*inputs[1153] -0.00698259*inputs[1154] -0.00698258*inputs[1155] +0.0155592*inputs[1156] +0.0231557*inputs[1157] +0.0400591*inputs[1158] +0.00140249*inputs[1159] +0.00140253*inputs[1160] -0.0010946*inputs[1161] +0.00140255*inputs[1162] -0.023256*inputs[1163] +0.00140246*inputs[1164] +0.00140249*inputs[1165] +0.0014025*inputs[1166] +0.0236732*inputs[1167] +0.0014025*inputs[1168] +0.00140248*inputs[1169] +0.00140246*inputs[1170] +0.00140247*inputs[1171] +0.00140248*inputs[1172] -0.00165632*inputs[1173] -0.00819118*inputs[1174] -0.00819119*inputs[1175] -0.00819119*inputs[1176] -0.00819119*inputs[1177] -0.012338*inputs[1178] -0.0123379*inputs[1179] -0.0123379*inputs[1180] -0.0123379*inputs[1181] +0.00140246*inputs[1182] +0.0014025*inputs[1183] +0.00140246*inputs[1184] +0.00140253*inputs[1185] +0.0336149*inputs[1186] +0.00140247*inputs[1187] +0.00140248*inputs[1188] +0.00140246*inputs[1189] -0.0297109*inputs[1190] +0.00140254*inputs[1191] -0.00354307*inputs[1192] +0.0325572*inputs[1193] -0.0267826*inputs[1194] +0.00140247*inputs[1195] +0.00140246*inputs[1196] -0.0302505*inputs[1197] -0.0197732*inputs[1198] -0.0197733*inputs[1199] +0.0238357*inputs[1200] +0.00140246*inputs[1201] +0.00140246*inputs[1202] +0.00140247*inputs[1203] +0.00140248*inputs[1204] +0.0220894*inputs[1205] -0.00763294*inputs[1206] -0.00763295*inputs[1207] +0.00140247*inputs[1208] +0.00140252*inputs[1209] +0.00140249*inputs[1210] +0.00140249*inputs[1211] +0.00140249*inputs[1212] +0.00140247*inputs[1213] +0.00140247*inputs[1214] +0.00140246*inputs[1215] +0.00140246*inputs[1216] +0.00140246*inputs[1217] +0.00140249*inputs[1218] -0.0208678*inputs[1219] -0.0208678*inputs[1220] -0.0208677*inputs[1221] +0.0128785*inputs[1222] +0.0128785*inputs[1223] +0.0128785*inputs[1224] +0.0196031*inputs[1225] +0.0196031*inputs[1226] +0.0014025*inputs[1227] +0.00140246*inputs[1228] +0.00140246*inputs[1229] +0.0054358*inputs[1230] +0.00543585*inputs[1231] +0.0313482*inputs[1232] +0.00140246*inputs[1233] +0.00140249*inputs[1234] +0.00140247*inputs[1235] -0.0315758*inputs[1236] +0.00140253*inputs[1237] +0.0165656*inputs[1238] +0.0168232*inputs[1239] +0.0168232*inputs[1240] +0.00140246*inputs[1241] +0.00140247*inputs[1242] +0.00140246*inputs[1243] +0.00140247*inputs[1244] +0.00140246*inputs[1245] +0.00711616*inputs[1246] +0.00711616*inputs[1247] +0.00711616*inputs[1248] -0.0142764*inputs[1249] +0.00140255*inputs[1250] +0.00140246*inputs[1251] +0.00758063*inputs[1252] +0.00140246*inputs[1253] +0.0181472*inputs[1254] +0.00601516*inputs[1255] +0.00601516*inputs[1256] +0.016317*inputs[1257] +0.0105844*inputs[1258] +0.0105844*inputs[1259] -0.00352085*inputs[1260] -0.00450623*inputs[1261] -0.00352087*inputs[1262] -0.00352087*inputs[1263] -0.00352083*inputs[1264] +0.0328207*inputs[1265] -0.0366868*inputs[1266] +0.00140246*inputs[1267] +0.00140246*inputs[1268] +0.00140252*inputs[1269] -0.0222108*inputs[1270] -0.01916*inputs[1271] +0.00738294*inputs[1272] +0.00738294*inputs[1273] -0.00974776*inputs[1274] -0.00974776*inputs[1275] +0.00140246*inputs[1276] +0.00140246*inputs[1277] -0.0224195*inputs[1278] +0.00140246*inputs[1279] +0.00140253*inputs[1280] -0.00450623*inputs[1281] +0.00140247*inputs[1282] +0.00140249*inputs[1283] +0.0343523*inputs[1284] +0.0127093*inputs[1285] +0.0463341*inputs[1286] +0.00140248*inputs[1287] +0.00140247*inputs[1288] +0.00140248*inputs[1289] +0.00849799*inputs[1290] +0.0119934*inputs[1291] +0.0156729*inputs[1292] +0.0319353*inputs[1293] +0.0119355*inputs[1294] +0.00610762*inputs[1295] +0.00140247*inputs[1296] +0.0119934*inputs[1297] -0.0316303*inputs[1298] +0.00140248*inputs[1299] +0.00140246*inputs[1300] +0.00140246*inputs[1301] +0.00140246*inputs[1302] -0.00443643*inputs[1303] +0.00140247*inputs[1304] -0.00982912*inputs[1305] -0.00982915*inputs[1306] -0.00982915*inputs[1307] +0.00140247*inputs[1308] -0.00895337*inputs[1309] +0.00140246*inputs[1310] +0.00140248*inputs[1311] +0.0014025*inputs[1312] +0.00407716*inputs[1313] +0.00407716*inputs[1314] +0.00140247*inputs[1315] +0.00140246*inputs[1316] +0.00508024*inputs[1317] +0.00140247*inputs[1318] +0.0163979*inputs[1319] -0.0308723*inputs[1320] +0.00140246*inputs[1321] +0.00633689*inputs[1322] +0.00140246*inputs[1323] +0.00140247*inputs[1324] -0.039069*inputs[1325] -0.00909234*inputs[1326] +0.00140246*inputs[1327] +0.0014025*inputs[1328] +0.00140247*inputs[1329] +0.00140253*inputs[1330] +0.0153253*inputs[1331] +0.00140246*inputs[1332] +0.00140246*inputs[1333] +0.00140247*inputs[1334] +0.00140246*inputs[1335] -0.0134416*inputs[1336] -0.0134416*inputs[1337] -0.0134416*inputs[1338] +0.0108818*inputs[1339] +0.0108818*inputs[1340] +0.00140246*inputs[1341] +0.00753051*inputs[1342] +0.00140246*inputs[1343] +0.00140246*inputs[1344] +0.00140247*inputs[1345] +0.017149*inputs[1346] +0.00140248*inputs[1347] +0.00140248*inputs[1348] +0.0014025*inputs[1349] +0.00140246*inputs[1350] +0.00140247*inputs[1351] +0.00140248*inputs[1352] -0.0124212*inputs[1353] -0.0124212*inputs[1354] +0.00140246*inputs[1355] +0.00140246*inputs[1356] -0.0346654*inputs[1357] -0.0376066*inputs[1358] +0.00140248*inputs[1359] +0.00140246*inputs[1360] +0.00140246*inputs[1361] +0.00140246*inputs[1362] +0.00140246*inputs[1363] -0.0229756*inputs[1364] -0.0229757*inputs[1365] -0.000470247*inputs[1366] -0.000470246*inputs[1367] -0.00047023*inputs[1368] -0.00047018*inputs[1369] -0.000470246*inputs[1370] +0.0231092*inputs[1371] +0.00681857*inputs[1372] +0.00681856*inputs[1373] -0.00835622*inputs[1374] +0.00669049*inputs[1375] +0.00669058*inputs[1376] +0.0125253*inputs[1377] +0.0495512*inputs[1378] -0.00195084*inputs[1379] +0.00140254*inputs[1380] +0.00140255*inputs[1381] +0.00140247*inputs[1382] +0.00140246*inputs[1383] +0.0345017*inputs[1384] +0.0165285*inputs[1385] +0.0165285*inputs[1386] +0.00140246*inputs[1387] +0.00140248*inputs[1388] +0.00140247*inputs[1389] +0.029899*inputs[1390] +0.00140248*inputs[1391] +0.00140247*inputs[1392] +0.00140246*inputs[1393] +0.00140248*inputs[1394] +0.00140246*inputs[1395] 
		combinations[5] = 0.049073 +0.078609*inputs[0] -0.015192*inputs[1] +0.223904*inputs[2] +0.161698*inputs[3] +0.153398*inputs[4] +0.0700588*inputs[5] +0.0107359*inputs[6] +0.0345054*inputs[7] +0.0481176*inputs[8] -0.00829133*inputs[9] +0.0254829*inputs[10] +0.0961649*inputs[11] +0.0294014*inputs[12] +0.10053*inputs[13] +0.17907*inputs[14] +0.0350974*inputs[15] +0.00686261*inputs[16] +0.0121953*inputs[17] +0.107335*inputs[18] -0.0990303*inputs[19] +0.00478374*inputs[20] +0.0156232*inputs[21] +0.0256021*inputs[22] +0.0218967*inputs[23] -0.00367989*inputs[24] -0.0234602*inputs[25] +0.126764*inputs[26] -0.0130453*inputs[27] +0.0119836*inputs[28] +0.00214093*inputs[29] +0.0140813*inputs[30] -0.0216618*inputs[31] -0.111294*inputs[32] +0.0975027*inputs[33] -0.0816969*inputs[34] +0.103989*inputs[35] +0.0912029*inputs[36] +0.00926091*inputs[37] -0.00780138*inputs[38] -0.126379*inputs[39] +0.0406019*inputs[40] -0.119902*inputs[41] +0.014539*inputs[42] -0.0420924*inputs[43] +0.034664*inputs[44] +0.0116005*inputs[45] -0.0198352*inputs[46] -0.108349*inputs[47] -0.0797899*inputs[48] +0.0901202*inputs[49] +0.00274077*inputs[50] +0.00318878*inputs[51] +0.00133048*inputs[52] +0.0103196*inputs[53] -0.00333734*inputs[54] -0.0471248*inputs[55] +0.0578042*inputs[56] +0.0329125*inputs[57] -0.106226*inputs[58] -0.00384304*inputs[59] -0.122103*inputs[60] -0.0304867*inputs[61] -0.0292644*inputs[62] -0.0366648*inputs[63] +0.0018765*inputs[64] -0.153292*inputs[65] -0.0819391*inputs[66] +0.0474349*inputs[67] -0.04005*inputs[68] +0.000957531*inputs[69] +0.0113769*inputs[70] -0.00266971*inputs[71] +0.0105252*inputs[72] -0.0112358*inputs[73] -0.0912731*inputs[74] +0.00877502*inputs[75] +0.100279*inputs[76] +0.0180291*inputs[77] +0.0172906*inputs[78] +0.0111114*inputs[79] -0.0415101*inputs[80] -0.0400244*inputs[81] +0.0103271*inputs[82] +0.0839558*inputs[83] +0.0240994*inputs[84] -0.0409866*inputs[85] +0.0435663*inputs[86] -0.017396*inputs[87] -0.00470212*inputs[88] -0.0154718*inputs[89] -0.0106489*inputs[90] -0.023502*inputs[91] -0.00264231*inputs[92] -0.0223937*inputs[93] -0.108224*inputs[94] -0.0627251*inputs[95] +0.00693012*inputs[96] +0.0214135*inputs[97] -0.0125*inputs[98] +0.0371318*inputs[99] +0.0066435*inputs[100] -0.0275646*inputs[101] -0.021601*inputs[102] -0.0130003*inputs[103] -0.0472833*inputs[104] +0.00308241*inputs[105] +0.204089*inputs[106] +0.00558118*inputs[107] -0.0828187*inputs[108] +0.00714685*inputs[109] +0.0455269*inputs[110] +0.0592574*inputs[111] -0.0199323*inputs[112] +0.0188307*inputs[113] -0.0628604*inputs[114] +0.0201933*inputs[115] -0.00380469*inputs[116] -0.016593*inputs[117] -0.0324825*inputs[118] -0.0141009*inputs[119] -0.0278178*inputs[120] +0.0802514*inputs[121] +0.0732294*inputs[122] -0.0813742*inputs[123] -0.0280471*inputs[124] -0.0124375*inputs[125] -0.00372106*inputs[126] -0.00361748*inputs[127] +0.00931325*inputs[128] +0.00720094*inputs[129] +0.0117445*inputs[130] +0.00871345*inputs[131] +0.0206374*inputs[132] +0.0116816*inputs[133] -0.0322613*inputs[134] -0.100721*inputs[135] -0.0744635*inputs[136] -0.010031*inputs[137] -0.0224886*inputs[138] -0.0843681*inputs[139] +0.00836702*inputs[140] +0.0218849*inputs[141] +0.0286101*inputs[142] +0.0385813*inputs[143] -0.0145194*inputs[144] +0.00873857*inputs[145] +0.0650498*inputs[146] -0.0201056*inputs[147] -0.0253626*inputs[148] -0.0374802*inputs[149] -0.0168678*inputs[150] -0.0145708*inputs[151] -0.0372202*inputs[152] -0.000896254*inputs[153] -0.0594926*inputs[154] -0.0157815*inputs[155] +0.00327287*inputs[156] -0.00824949*inputs[157] -0.0207318*inputs[158] +0.0638655*inputs[159] -0.00357585*inputs[160] -0.014164*inputs[161] +0.0020847*inputs[162] -0.0223086*inputs[163] -0.00772938*inputs[164] -0.00361564*inputs[165] +0.0195754*inputs[166] +0.00613953*inputs[167] +0.00231627*inputs[168] -0.0218282*inputs[169] -0.0124337*inputs[170] -0.0367395*inputs[171] +0.00727948*inputs[172] +0.0205711*inputs[173] -0.0184361*inputs[174] +0.0638555*inputs[175] +0.0146423*inputs[176] -0.00765994*inputs[177] -0.0692945*inputs[178] +0.0468989*inputs[179] -0.0568826*inputs[180] -0.015189*inputs[181] +0.00363538*inputs[182] +0.00569377*inputs[183] -0.0106129*inputs[184] -0.000986589*inputs[185] -0.0119247*inputs[186] -0.0115719*inputs[187] +0.0152277*inputs[188] -0.00918199*inputs[189] -0.0235789*inputs[190] -0.0117264*inputs[191] +0.0343323*inputs[192] -0.00372489*inputs[193] +0.0493431*inputs[194] +0.0185928*inputs[195] +0.0194361*inputs[196] -0.0363885*inputs[197] +0.0352475*inputs[198] -0.00743316*inputs[199] -0.00331766*inputs[200] +0.00466475*inputs[201] -0.00418729*inputs[202] -0.0126566*inputs[203] +0.0269635*inputs[204] -0.0407859*inputs[205] +0.0217004*inputs[206] -0.0253831*inputs[207] -0.0176717*inputs[208] -0.0537293*inputs[209] +0.00101775*inputs[210] -0.00590686*inputs[211] -0.0279136*inputs[212] -0.00304862*inputs[213] +0.0169487*inputs[214] +0.0559044*inputs[215] +0.0263659*inputs[216] -0.0217067*inputs[217] +0.0197995*inputs[218] -0.00704236*inputs[219] +0.016904*inputs[220] -0.0189376*inputs[221] +0.013892*inputs[222] +0.00262149*inputs[223] -0.000401938*inputs[224] -0.0539076*inputs[225] -0.000113934*inputs[226] -0.00763517*inputs[227] +0.00350621*inputs[228] +0.0153845*inputs[229] +0.0197783*inputs[230] +0.00441352*inputs[231] +0.015296*inputs[232] +0.00968948*inputs[233] +0.00172578*inputs[234] +0.0405722*inputs[235] -0.000695788*inputs[236] +0.0223209*inputs[237] -0.011437*inputs[238] -0.0405246*inputs[239] -0.00347524*inputs[240] -0.00187289*inputs[241] -0.00503267*inputs[242] -0.0171107*inputs[243] -0.0137802*inputs[244] +0.036771*inputs[245] -0.0133289*inputs[246] -0.00417855*inputs[247] -0.0473221*inputs[248] -0.00775627*inputs[249] -0.00912517*inputs[250] -0.0283439*inputs[251] -0.0117169*inputs[252] -0.0357178*inputs[253] -0.0122766*inputs[254] -0.00765082*inputs[255] +0.0113887*inputs[256] -0.00283699*inputs[257] +0.00103612*inputs[258] -0.00642179*inputs[259] -0.00310675*inputs[260] +0.0293561*inputs[261] +0.0119636*inputs[262] -0.000893306*inputs[263] +0.00446844*inputs[264] -0.0104533*inputs[265] +0.0228209*inputs[266] -0.0585673*inputs[267] -0.0235829*inputs[268] +0.00977238*inputs[269] -0.00869853*inputs[270] -0.0530412*inputs[271] -0.00490038*inputs[272] +0.00697728*inputs[273] -0.00310673*inputs[274] -0.00310675*inputs[275] +0.0100175*inputs[276] -0.00359812*inputs[277] -0.0503537*inputs[278] +0.00228483*inputs[279] +0.0229492*inputs[280] -0.00242937*inputs[281] +0.01997*inputs[282] +0.0300496*inputs[283] +0.0202772*inputs[284] +0.102516*inputs[285] -0.011711*inputs[286] -0.0286766*inputs[287] -0.016396*inputs[288] +0.0194743*inputs[289] +0.0258654*inputs[290] -0.00193735*inputs[291] -0.00778018*inputs[292] -0.0165892*inputs[293] +0.0181526*inputs[294] +0.00418784*inputs[295] -0.0158429*inputs[296] +0.00102844*inputs[297] +0.0201606*inputs[298] +0.0476989*inputs[299] +0.0315489*inputs[300] +0.00954982*inputs[301] +0.00526486*inputs[302] -0.0866204*inputs[303] +0.005725*inputs[304] -0.0431955*inputs[305] -0.0182238*inputs[306] -0.0388649*inputs[307] -0.0174182*inputs[308] -0.0204803*inputs[309] -0.00283699*inputs[310] -0.00868114*inputs[311] -0.0495942*inputs[312] +0.0112552*inputs[313] +0.0390637*inputs[314] +0.00348352*inputs[315] -0.0067616*inputs[316] +0.0094889*inputs[317] +0.00134228*inputs[318] +0.0274894*inputs[319] -0.0285179*inputs[320] -0.00685951*inputs[321] -0.0055829*inputs[322] -0.0162122*inputs[323] -0.00319464*inputs[324] -0.000403547*inputs[325] -0.00631858*inputs[326] +0.00431515*inputs[327] -0.000774675*inputs[328] -0.00268908*inputs[329] -0.0130627*inputs[330] +0.0180981*inputs[331] +0.0287798*inputs[332] -0.00558293*inputs[333] -0.0257966*inputs[334] -0.0230812*inputs[335] -0.0113832*inputs[336] +0.018051*inputs[337] -0.0417345*inputs[338] -0.00639568*inputs[339] -0.0337384*inputs[340] +0.0477207*inputs[341] +0.000198812*inputs[342] +0.0101246*inputs[343] -0.02045*inputs[344] +0.0288874*inputs[345] -0.0166854*inputs[346] -0.00208161*inputs[347] -0.0191275*inputs[348] +0.00670518*inputs[349] -0.000268295*inputs[350] +0.0392994*inputs[351] -0.00665885*inputs[352] +0.00987425*inputs[353] -0.0140244*inputs[354] +0.00871613*inputs[355] +0.00387348*inputs[356] +0.0108602*inputs[357] -0.00268908*inputs[358] +0.00754633*inputs[359] +0.0142917*inputs[360] +0.000140764*inputs[361] -0.00297721*inputs[362] -0.0127123*inputs[363] +0.0123968*inputs[364] -0.0172944*inputs[365] +0.000737396*inputs[366] -0.0220406*inputs[367] +0.0194271*inputs[368] -0.0389805*inputs[369] -0.00268914*inputs[370] -0.00268909*inputs[371] -0.00921549*inputs[372] -0.0120094*inputs[373] +0.0183423*inputs[374] -0.010103*inputs[375] +0.00916282*inputs[376] +0.034896*inputs[377] +0.0118236*inputs[378] +0.017218*inputs[379] +0.00275479*inputs[380] +0.0415263*inputs[381] -0.0234043*inputs[382] -0.0112866*inputs[383] -0.0758375*inputs[384] -0.00539583*inputs[385] -0.00552468*inputs[386] +0.00962632*inputs[387] -0.00268912*inputs[388] -0.0123901*inputs[389] -0.00824081*inputs[390] +0.00805129*inputs[391] -0.00268915*inputs[392] +0.0300694*inputs[393] +0.0149653*inputs[394] -0.0417133*inputs[395] +0.0035884*inputs[396] +0.0102649*inputs[397] +0.0137341*inputs[398] -0.0116253*inputs[399] +0.023765*inputs[400] +0.0278584*inputs[401] -0.010915*inputs[402] +0.0156058*inputs[403] -0.00651506*inputs[404] +0.000118678*inputs[405] -0.0636649*inputs[406] +0.000181695*inputs[407] -0.028167*inputs[408] -0.0115122*inputs[409] +0.000715777*inputs[410] -0.0452946*inputs[411] +0.00387647*inputs[412] -0.00268908*inputs[413] -0.0199513*inputs[414] +0.0039794*inputs[415] -0.0246023*inputs[416] +0.00083265*inputs[417] -0.0175157*inputs[418] -0.0117137*inputs[419] -0.00078406*inputs[420] -0.0304358*inputs[421] -0.00268908*inputs[422] -0.0248364*inputs[423] -0.0139213*inputs[424] -0.0186044*inputs[425] -0.0106595*inputs[426] -0.0357987*inputs[427] +0.0119721*inputs[428] +0.00862368*inputs[429] +0.0093039*inputs[430] +0.0138863*inputs[431] -0.0475646*inputs[432] +0.0107651*inputs[433] -0.0189917*inputs[434] -0.00914788*inputs[435] -0.00547421*inputs[436] +0.00330474*inputs[437] +0.000866031*inputs[438] -0.0313643*inputs[439] -0.000111582*inputs[440] -0.0026891*inputs[441] -0.0216566*inputs[442] -0.0159325*inputs[443] -0.00672728*inputs[444] -0.0337315*inputs[445] -0.075327*inputs[446] +0.0258435*inputs[447] -0.0180624*inputs[448] -0.006261*inputs[449] -0.00219446*inputs[450] -0.0384542*inputs[451] -0.0121979*inputs[452] -0.00975151*inputs[453] +0.0162727*inputs[454] -0.0247184*inputs[455] -0.0247183*inputs[456] +0.00445906*inputs[457] +0.00399112*inputs[458] +0.0552461*inputs[459] -0.00219444*inputs[460] -0.0254022*inputs[461] -0.0231748*inputs[462] -0.00155082*inputs[463] -0.0119169*inputs[464] -0.00219444*inputs[465] +0.0160012*inputs[466] -0.0588545*inputs[467] +0.00747361*inputs[468] +0.00798625*inputs[469] +0.00145033*inputs[470] +0.0206072*inputs[471] +0.034364*inputs[472] -0.0134912*inputs[473] +0.00862311*inputs[474] +0.0137944*inputs[475] -0.0146783*inputs[476] -0.0173489*inputs[477] +0.00652878*inputs[478] -0.027157*inputs[479] -0.0230402*inputs[480] +0.0225601*inputs[481] -0.00420436*inputs[482] -0.0117784*inputs[483] -0.00219444*inputs[484] -0.00306088*inputs[485] -0.00923912*inputs[486] -0.00219444*inputs[487] +0.024773*inputs[488] -0.0127588*inputs[489] -0.0911681*inputs[490] -0.00218229*inputs[491] +0.00687754*inputs[492] -0.0218399*inputs[493] +0.00317469*inputs[494] +0.00534921*inputs[495] -0.0255352*inputs[496] -0.0297692*inputs[497] -0.0171263*inputs[498] +0.00370849*inputs[499] +0.0400412*inputs[500] +0.0186238*inputs[501] -0.0235664*inputs[502] -0.00219448*inputs[503] -0.0305518*inputs[504] -0.00219446*inputs[505] +0.00583896*inputs[506] +0.028277*inputs[507] +0.00790272*inputs[508] -0.0254999*inputs[509] +0.0173389*inputs[510] +0.014806*inputs[511] -0.0396235*inputs[512] -0.0155869*inputs[513] -0.00219449*inputs[514] -0.0390099*inputs[515] -0.0174624*inputs[516] -0.00219444*inputs[517] -0.00296722*inputs[518] -0.0124752*inputs[519] -0.0194742*inputs[520] -0.0101837*inputs[521] -0.0100259*inputs[522] -0.0100259*inputs[523] -0.0402192*inputs[524] +0.0214067*inputs[525] -0.0132208*inputs[526] -0.0104725*inputs[527] +0.00362222*inputs[528] -0.00219448*inputs[529] -0.0070601*inputs[530] +0.00864085*inputs[531] +0.00864086*inputs[532] -0.0021945*inputs[533] +0.013663*inputs[534] +0.010335*inputs[535] -0.0112048*inputs[536] -0.00630292*inputs[537] -0.00899049*inputs[538] +0.016138*inputs[539] +0.016138*inputs[540] -0.00227972*inputs[541] -0.00271021*inputs[542] +0.00168451*inputs[543] -0.0393212*inputs[544] -0.00219446*inputs[545] -0.00219444*inputs[546] +0.00501357*inputs[547] -0.0116426*inputs[548] -0.00219444*inputs[549] -0.00219446*inputs[550] -0.0314974*inputs[551] -0.0212399*inputs[552] +0.010335*inputs[553] -0.0056528*inputs[554] -0.00219445*inputs[555] +0.00435057*inputs[556] -0.0333431*inputs[557] -0.00352138*inputs[558] -0.0454154*inputs[559] -0.00415234*inputs[560] +0.0235733*inputs[561] -0.000607435*inputs[562] -0.0403761*inputs[563] +0.039232*inputs[564] +0.0129418*inputs[565] -0.00436341*inputs[566] -0.0233257*inputs[567] -0.00219445*inputs[568] +0.00924424*inputs[569] +0.00570181*inputs[570] -0.027142*inputs[571] -0.0128665*inputs[572] +0.00603778*inputs[573] +0.00950601*inputs[574] -0.0139438*inputs[575] +0.00215909*inputs[576] -0.00218231*inputs[577] -0.00219444*inputs[578] -0.0238898*inputs[579] +0.0188974*inputs[580] -0.00155091*inputs[581] -0.00622249*inputs[582] -0.0144641*inputs[583] +0.00589314*inputs[584] -0.027688*inputs[585] +0.00446958*inputs[586] +0.0135234*inputs[587] -0.00219449*inputs[588] -0.00219444*inputs[589] -0.0243086*inputs[590] +0.00790272*inputs[591] -0.0318049*inputs[592] -0.00494241*inputs[593] -0.0101011*inputs[594] +0.00833508*inputs[595] +0.0221505*inputs[596] +0.00112789*inputs[597] -0.00514736*inputs[598] +0.01015*inputs[599] -0.00034015*inputs[600] -0.00167414*inputs[601] -0.00219446*inputs[602] +0.0244024*inputs[603] +0.0202964*inputs[604] -0.0257774*inputs[605] -0.030499*inputs[606] -0.00219449*inputs[607] -0.0393489*inputs[608] -0.0132453*inputs[609] +0.00722708*inputs[610] +0.0206628*inputs[611] -0.00675832*inputs[612] -0.000497106*inputs[613] -0.00219444*inputs[614] -0.00219444*inputs[615] -0.0431917*inputs[616] +0.0160274*inputs[617] -0.00849707*inputs[618] -0.00436342*inputs[619] +0.00220823*inputs[620] -0.00125183*inputs[621] -0.0110086*inputs[622] +0.00282475*inputs[623] -0.014576*inputs[624] +0.0119543*inputs[625] -0.00219447*inputs[626] -0.0125859*inputs[627] +0.0135234*inputs[628] +0.00716099*inputs[629] -0.0524639*inputs[630] +0.0274834*inputs[631] -0.00296721*inputs[632] +0.0110596*inputs[633] +0.000603403*inputs[634] +0.00197344*inputs[635] +0.00060337*inputs[636] +0.0261039*inputs[637] +0.0225756*inputs[638] +0.0127846*inputs[639] +0.0155446*inputs[640] -0.0174243*inputs[641] -0.00490581*inputs[642] -0.00219444*inputs[643] -0.00375994*inputs[644] -0.00831505*inputs[645] -0.0121979*inputs[646] +0.00131978*inputs[647] -0.00155082*inputs[648] -0.00381252*inputs[649] -0.00697707*inputs[650] -0.0122329*inputs[651] -0.0122329*inputs[652] -0.0122328*inputs[653] +0.00610279*inputs[654] -0.00155082*inputs[655] -0.00155091*inputs[656] -0.00155084*inputs[657] -0.00155085*inputs[658] -0.00155081*inputs[659] -0.0015508*inputs[660] -0.0015508*inputs[661] -0.00991702*inputs[662] -0.0109527*inputs[663] -0.0109527*inputs[664] -0.0015508*inputs[665] +0.0113407*inputs[666] -0.018075*inputs[667] -0.018075*inputs[668] -0.0310885*inputs[669] +0.00852723*inputs[670] -0.0159314*inputs[671] -0.0015508*inputs[672] +0.00484479*inputs[673] +0.00484476*inputs[674] +0.0048448*inputs[675] -0.0115085*inputs[676] -0.0115085*inputs[677] -0.0115085*inputs[678] -0.0228007*inputs[679] +0.0225666*inputs[680] +0.0225666*inputs[681] +0.0225666*inputs[682] -0.00155084*inputs[683] -0.00155082*inputs[684] -0.00155081*inputs[685] -0.00155083*inputs[686] -0.0015508*inputs[687] -0.0104556*inputs[688] -0.0104556*inputs[689] -0.0015508*inputs[690] -0.00155082*inputs[691] +0.00980455*inputs[692] +0.00980455*inputs[693] +0.00980455*inputs[694] +0.00980455*inputs[695] -0.00155081*inputs[696] -0.00155084*inputs[697] -0.00155081*inputs[698] +0.0016872*inputs[699] +0.0016872*inputs[700] -0.0126206*inputs[701] -0.00655805*inputs[702] -0.00655806*inputs[703] -0.00655805*inputs[704] -0.00655806*inputs[705] +0.00656667*inputs[706] -0.00155083*inputs[707] -0.00155088*inputs[708] -0.00155084*inputs[709] -0.00155088*inputs[710] -0.00155081*inputs[711] -0.00155089*inputs[712] -0.0603653*inputs[713] -0.024658*inputs[714] -0.0424791*inputs[715] +0.00716412*inputs[716] +0.00716412*inputs[717] -0.0015508*inputs[718] -0.0015508*inputs[719] -0.010812*inputs[720] -0.010812*inputs[721] +0.000692484*inputs[722] -0.00261248*inputs[723] -0.00155081*inputs[724] -0.00155084*inputs[725] -0.00155082*inputs[726] -0.00155081*inputs[727] -0.00155081*inputs[728] -0.0334562*inputs[729] +0.00215909*inputs[730] +0.00215907*inputs[731] +0.00215905*inputs[732] +0.00215908*inputs[733] +0.00215908*inputs[734] -0.00155081*inputs[735] -0.00155085*inputs[736] -0.00155081*inputs[737] +0.0169133*inputs[738] -0.00155081*inputs[739] -0.0015508*inputs[740] -0.00155084*inputs[741] +0.0151189*inputs[742] +0.0151189*inputs[743] +0.00771608*inputs[744] +0.0291599*inputs[745] +0.0245883*inputs[746] +0.0245883*inputs[747] -0.0120902*inputs[748] -0.0120902*inputs[749] +0.00651597*inputs[750] +0.00651598*inputs[751] +0.0117666*inputs[752] -0.00155083*inputs[753] -0.00155081*inputs[754] +0.00460071*inputs[755] +0.00460064*inputs[756] +0.00460072*inputs[757] +0.00460072*inputs[758] +0.00984516*inputs[759] +0.00984516*inputs[760] -0.00155081*inputs[761] -0.0020494*inputs[762] -0.00204938*inputs[763] -0.00204943*inputs[764] -0.00204941*inputs[765] -0.00204939*inputs[766] -0.00204938*inputs[767] -0.0083742*inputs[768] -0.00837419*inputs[769] -0.0015508*inputs[770] +0.0121859*inputs[771] +0.0121859*inputs[772] -0.00155082*inputs[773] -0.00155083*inputs[774] -0.00155082*inputs[775] -0.00155082*inputs[776] -0.0015508*inputs[777] -0.0015508*inputs[778] -0.00155082*inputs[779] -0.00155081*inputs[780] -0.0496417*inputs[781] -0.0248071*inputs[782] -0.0248069*inputs[783] -0.00851706*inputs[784] -0.00851707*inputs[785] -0.0222807*inputs[786] -0.0015508*inputs[787] -0.00155085*inputs[788] -0.00155082*inputs[789] +0.0139294*inputs[790] -0.00155081*inputs[791] +0.00872051*inputs[792] -0.00155084*inputs[793] -0.00155084*inputs[794] +0.00954517*inputs[795] +0.00954517*inputs[796] +0.0231914*inputs[797] -0.00155084*inputs[798] -0.00155081*inputs[799] -0.00155081*inputs[800] +0.0016626*inputs[801] +0.00166267*inputs[802] -0.0127582*inputs[803] -0.0015508*inputs[804] -0.00651134*inputs[805] -0.00155081*inputs[806] +0.00615023*inputs[807] -0.00155081*inputs[808] -0.00155087*inputs[809] -0.00155085*inputs[810] -0.00155084*inputs[811] -0.00651133*inputs[812] +0.0352898*inputs[813] +0.00280683*inputs[814] +0.00280684*inputs[815] -0.0015508*inputs[816] +0.00615026*inputs[817] -0.0015508*inputs[818] -0.0015509*inputs[819] -0.0015508*inputs[820] -0.00155084*inputs[821] -0.0316152*inputs[822] -0.0269926*inputs[823] -0.00392119*inputs[824] -0.00155081*inputs[825] -0.00155087*inputs[826] +0.00615024*inputs[827] -0.00392119*inputs[828] +0.0312749*inputs[829] +0.0161598*inputs[830] +0.00679303*inputs[831] +0.00679297*inputs[832] +0.0116835*inputs[833] -0.00392121*inputs[834] +0.00679303*inputs[835] +0.00629637*inputs[836] +0.00770058*inputs[837] -0.00155081*inputs[838] +0.0135989*inputs[839] -0.00155082*inputs[840] -0.000936952*inputs[841] +0.0180232*inputs[842] -0.00964442*inputs[843] -0.00964442*inputs[844] -0.00964442*inputs[845] -0.00964442*inputs[846] -0.00964442*inputs[847] +0.0238176*inputs[848] -0.0144718*inputs[849] -0.0144718*inputs[850] +0.0238176*inputs[851] -0.0239459*inputs[852] -0.0128435*inputs[853] +0.0142313*inputs[854] +0.00852712*inputs[855] -0.0015508*inputs[856] -0.0015508*inputs[857] -0.0015508*inputs[858] +0.00174548*inputs[859] -0.0015508*inputs[860] -0.00155089*inputs[861] -0.00155082*inputs[862] -0.00155083*inputs[863] -0.00155092*inputs[864] +0.0184756*inputs[865] -0.00342628*inputs[866] -0.0128435*inputs[867] -0.0159488*inputs[868] -0.0015508*inputs[869] -0.0078637*inputs[870] -0.00786373*inputs[871] -0.00155091*inputs[872] -0.00746036*inputs[873] -0.00746035*inputs[874] -0.00746038*inputs[875] -0.00155081*inputs[876] +0.00629642*inputs[877] -0.0222894*inputs[878] -0.0239745*inputs[879] -0.0222894*inputs[880] +0.0120178*inputs[881] -0.0365401*inputs[882] -0.00155091*inputs[883] -0.00155082*inputs[884] +0.00330007*inputs[885] +0.00330006*inputs[886] -0.00155081*inputs[887] -0.00155082*inputs[888] -0.0015508*inputs[889] -0.00155081*inputs[890] -0.00155089*inputs[891] -0.00155084*inputs[892] -0.0015508*inputs[893] -0.00155083*inputs[894] -0.00155081*inputs[895] -0.00155085*inputs[896] -0.00689598*inputs[897] -0.00155085*inputs[898] -0.00146878*inputs[899] -0.00689597*inputs[900] +0.00629636*inputs[901] +0.00806081*inputs[902] -0.0191204*inputs[903] -0.00155081*inputs[904] +0.0080608*inputs[905] +0.00806081*inputs[906] -0.000979987*inputs[907] -0.000979969*inputs[908] -0.00155082*inputs[909] -0.0015508*inputs[910] -0.0015508*inputs[911] +0.0273396*inputs[912] -0.0190522*inputs[913] -0.00155085*inputs[914] -0.0015508*inputs[915] -0.0191203*inputs[916] -0.0013818*inputs[917] -0.0015508*inputs[918] -0.00138181*inputs[919] -0.00138184*inputs[920] -0.0191203*inputs[921] -0.0375862*inputs[922] +0.0135517*inputs[923] -0.0125668*inputs[924] -0.0015508*inputs[925] -0.00155087*inputs[926] -0.00155081*inputs[927] -0.00155082*inputs[928] -0.0346266*inputs[929] -0.00155083*inputs[930] -0.0015508*inputs[931] -0.0308603*inputs[932] -0.0117784*inputs[933] +0.00893132*inputs[934] -0.00155083*inputs[935] -0.0117784*inputs[936] -0.00615225*inputs[937] +0.0089314*inputs[938] -0.0159042*inputs[939] -0.0043917*inputs[940] -0.0043917*inputs[941] -0.0043917*inputs[942] -0.00439169*inputs[943] +0.000867997*inputs[944] -0.0180752*inputs[945] +0.000868099*inputs[946] +0.0245529*inputs[947] -0.0015508*inputs[948] +0.00786894*inputs[949] +0.00786894*inputs[950] -0.0180752*inputs[951] -0.0200293*inputs[952] -0.00155083*inputs[953] -0.00155084*inputs[954] +0.00980966*inputs[955] +0.00980967*inputs[956] +0.00980967*inputs[957] +0.00980967*inputs[958] -0.0199411*inputs[959] -0.00155081*inputs[960] -0.00155081*inputs[961] +0.0105612*inputs[962] +0.0105612*inputs[963] +0.00451089*inputs[964] +0.0307265*inputs[965] -0.0015508*inputs[966] -0.0015508*inputs[967] -0.0015508*inputs[968] -0.0316151*inputs[969] -0.00155082*inputs[970] +0.0181841*inputs[971] -0.0015508*inputs[972] -0.00155082*inputs[973] -0.00155082*inputs[974] -0.00155084*inputs[975] -0.0502717*inputs[976] -0.0181209*inputs[977] +0.0130293*inputs[978] -0.00155085*inputs[979] -0.0209862*inputs[980] -0.00155082*inputs[981] +0.0116099*inputs[982] +0.0116099*inputs[983] -0.00155085*inputs[984] -0.00155081*inputs[985] -0.0015509*inputs[986] +0.000829653*inputs[987] -0.00155083*inputs[988] -0.00155089*inputs[989] -0.00155081*inputs[990] -0.00796441*inputs[991] -0.0159795*inputs[992] -0.00155084*inputs[993] +0.00340152*inputs[994] -0.00155084*inputs[995] -0.0207128*inputs[996] +0.0313889*inputs[997] -0.00796441*inputs[998] -0.00155081*inputs[999] -0.0015508*inputs[1000] -0.00155084*inputs[1001] -0.0108751*inputs[1002] -0.00572487*inputs[1003] -0.00796442*inputs[1004] -0.00155084*inputs[1005] -0.00155081*inputs[1006] -0.0015508*inputs[1007] -0.0229753*inputs[1008] +0.0133329*inputs[1009] +0.0133329*inputs[1010] +0.0133329*inputs[1011] -0.0015508*inputs[1012] -0.0015508*inputs[1013] -0.00155083*inputs[1014] -0.00155081*inputs[1015] -0.00838903*inputs[1016] -0.00838904*inputs[1017] +0.0158982*inputs[1018] +0.0158982*inputs[1019] -0.0226571*inputs[1020] -0.00081537*inputs[1021] -0.00155084*inputs[1022] -0.00155082*inputs[1023] -0.00796441*inputs[1024] -0.003621*inputs[1025] -0.00362098*inputs[1026] -0.00362098*inputs[1027] -0.00155081*inputs[1028] -0.00155082*inputs[1029] -0.00155084*inputs[1030] -0.00796441*inputs[1031] +0.00667106*inputs[1032] +0.0136215*inputs[1033] +0.00667105*inputs[1034] -0.0156797*inputs[1035] -0.0156797*inputs[1036] -0.0156797*inputs[1037] -0.0156797*inputs[1038] -0.0156797*inputs[1039] -0.0015508*inputs[1040] -0.0015508*inputs[1041] -0.00155087*inputs[1042] +0.0111285*inputs[1043] -0.0165241*inputs[1044] -0.0165241*inputs[1045] -0.0165241*inputs[1046] +0.0111285*inputs[1047] +0.0104242*inputs[1048] -0.0015508*inputs[1049] -0.0166919*inputs[1050] +0.0104242*inputs[1051] +0.0111285*inputs[1052] -0.0015508*inputs[1053] +0.000848402*inputs[1054] +0.000848434*inputs[1055] +0.000848423*inputs[1056] +0.000848399*inputs[1057] -0.0160827*inputs[1058] +0.0332956*inputs[1059] -0.0015508*inputs[1060] -0.00155084*inputs[1061] -0.00155085*inputs[1062] +0.0146178*inputs[1063] +0.0333455*inputs[1064] -0.0015508*inputs[1065] -0.011157*inputs[1066] -0.011157*inputs[1067] -0.0015508*inputs[1068] -0.0015508*inputs[1069] -0.00155081*inputs[1070] -0.0015508*inputs[1071] -0.0015508*inputs[1072] -0.00155083*inputs[1073] -0.0015508*inputs[1074] -0.00155081*inputs[1075] -0.0015508*inputs[1076] -0.0015508*inputs[1077] +0.00477806*inputs[1078] +0.0242059*inputs[1079] +0.0185869*inputs[1080] +0.00477808*inputs[1081] +0.00477808*inputs[1082] -0.00461665*inputs[1083] -0.00461665*inputs[1084] -0.00461664*inputs[1085] -0.00155082*inputs[1086] -0.00155082*inputs[1087] +0.00477808*inputs[1088] -0.0361328*inputs[1089] -0.00155082*inputs[1090] +0.0104242*inputs[1091] -0.0181757*inputs[1092] -0.0015508*inputs[1093] +0.0101402*inputs[1094] -0.00155084*inputs[1095] +0.0104242*inputs[1096] -0.00155081*inputs[1097] -0.00155081*inputs[1098] +0.000113369*inputs[1099] +0.000113326*inputs[1100] -0.00155083*inputs[1101] -0.00155081*inputs[1102] +0.0110175*inputs[1103] +0.0189142*inputs[1104] -0.00155085*inputs[1105] -0.00155081*inputs[1106] -0.00155081*inputs[1107] -0.0015508*inputs[1108] -0.0143792*inputs[1109] +0.0039321*inputs[1110] +0.0286091*inputs[1111] -0.0162393*inputs[1112] -0.0015508*inputs[1113] -0.0140624*inputs[1114] -0.00155082*inputs[1115] -0.0102762*inputs[1116] -0.0289374*inputs[1117] +0.0118219*inputs[1118] -0.00155083*inputs[1119] -0.0015508*inputs[1120] -0.00155088*inputs[1121] -0.0015508*inputs[1122] -0.0015508*inputs[1123] -0.00155085*inputs[1124] -0.0015508*inputs[1125] -0.00580696*inputs[1126] -0.00580696*inputs[1127] -0.00155081*inputs[1128] -0.00155082*inputs[1129] -0.0215093*inputs[1130] -0.0368142*inputs[1131] +0.0251335*inputs[1132] -0.00848454*inputs[1133] -0.00848455*inputs[1134] -0.0104446*inputs[1135] -0.0104446*inputs[1136] -0.0104446*inputs[1137] -0.0162945*inputs[1138] -0.0015508*inputs[1139] -0.00155081*inputs[1140] +0.00603864*inputs[1141] -0.0140624*inputs[1142] -0.00264312*inputs[1143] -0.0026431*inputs[1144] -0.00264311*inputs[1145] -0.00264312*inputs[1146] +0.0203983*inputs[1147] +0.0203983*inputs[1148] +0.00856641*inputs[1149] -0.00155081*inputs[1150] -0.00155084*inputs[1151] -0.00155081*inputs[1152] +0.00657948*inputs[1153] +0.00657948*inputs[1154] +0.00657948*inputs[1155] -0.0106212*inputs[1156] -0.0220184*inputs[1157] -0.0393232*inputs[1158] -0.0015508*inputs[1159] -0.0015509*inputs[1160] +0.00176171*inputs[1161] -0.0015508*inputs[1162] +0.022934*inputs[1163] -0.00155081*inputs[1164] -0.0015508*inputs[1165] -0.00155084*inputs[1166] -0.0239413*inputs[1167] -0.0015508*inputs[1168] -0.00155081*inputs[1169] -0.00155091*inputs[1170] -0.00155081*inputs[1171] -0.00155091*inputs[1172] +0.00240398*inputs[1173] +0.00850667*inputs[1174] +0.00850667*inputs[1175] +0.00850667*inputs[1176] +0.00850667*inputs[1177] +0.011838*inputs[1178] +0.0118381*inputs[1179] +0.0118381*inputs[1180] +0.0118381*inputs[1181] -0.00155084*inputs[1182] -0.00155081*inputs[1183] -0.0015508*inputs[1184] -0.00155083*inputs[1185] -0.033046*inputs[1186] -0.00155081*inputs[1187] -0.00155085*inputs[1188] -0.0015508*inputs[1189] +0.0328289*inputs[1190] -0.00155081*inputs[1191] +0.00467244*inputs[1192] -0.0315443*inputs[1193] +0.0263239*inputs[1194] -0.00155091*inputs[1195] -0.00155088*inputs[1196] +0.02864*inputs[1197] +0.0196221*inputs[1198] +0.0196221*inputs[1199] -0.0231841*inputs[1200] -0.0015508*inputs[1201] -0.00155081*inputs[1202] -0.0015508*inputs[1203] -0.00155084*inputs[1204] -0.0205106*inputs[1205] +0.00767917*inputs[1206] +0.0076792*inputs[1207] -0.00155084*inputs[1208] -0.0015509*inputs[1209] -0.00155083*inputs[1210] -0.0015508*inputs[1211] -0.0015508*inputs[1212] -0.00155084*inputs[1213] -0.00155081*inputs[1214] -0.00155085*inputs[1215] -0.00155086*inputs[1216] -0.00155091*inputs[1217] -0.00155081*inputs[1218] +0.0214203*inputs[1219] +0.0214203*inputs[1220] +0.0214202*inputs[1221] -0.0127176*inputs[1222] -0.0127176*inputs[1223] -0.0127176*inputs[1224] -0.0188063*inputs[1225] -0.0188063*inputs[1226] -0.00155087*inputs[1227] -0.00155081*inputs[1228] -0.00155081*inputs[1229] -0.00556552*inputs[1230] -0.00556552*inputs[1231] -0.0316152*inputs[1232] -0.00155083*inputs[1233] -0.0015508*inputs[1234] -0.00155084*inputs[1235] +0.0308178*inputs[1236] -0.0015508*inputs[1237] -0.0149058*inputs[1238] -0.0170021*inputs[1239] -0.0170021*inputs[1240] -0.00155082*inputs[1241] -0.00155082*inputs[1242] -0.0015508*inputs[1243] -0.00155082*inputs[1244] -0.00155081*inputs[1245] -0.00719522*inputs[1246] -0.00719524*inputs[1247] -0.00719523*inputs[1248] +0.0134451*inputs[1249] -0.0015508*inputs[1250] -0.00155081*inputs[1251] -0.00783098*inputs[1252] -0.00155083*inputs[1253] -0.0171712*inputs[1254] -0.00543515*inputs[1255] -0.00543514*inputs[1256] -0.0166658*inputs[1257] -0.0102023*inputs[1258] -0.0102023*inputs[1259] +0.00341654*inputs[1260] +0.0042523*inputs[1261] +0.00341655*inputs[1262] +0.00341655*inputs[1263] +0.00341653*inputs[1264] -0.0321729*inputs[1265] +0.0361133*inputs[1266] -0.0015508*inputs[1267] -0.00155082*inputs[1268] -0.0015508*inputs[1269] +0.021476*inputs[1270] +0.0185373*inputs[1271] -0.0073938*inputs[1272] -0.00739378*inputs[1273] +0.00977556*inputs[1274] +0.00977554*inputs[1275] -0.00155083*inputs[1276] -0.0015508*inputs[1277] +0.0214082*inputs[1278] -0.0015508*inputs[1279] -0.0015508*inputs[1280] +0.0042523*inputs[1281] -0.00155081*inputs[1282] -0.00155081*inputs[1283] -0.0152936*inputs[1284] -0.0122793*inputs[1285] -0.0470052*inputs[1286] -0.00155084*inputs[1287] -0.00155084*inputs[1288] -0.00155083*inputs[1289] -0.00817103*inputs[1290] -0.0127093*inputs[1291] -0.0156908*inputs[1292] -0.0311006*inputs[1293] -0.012294*inputs[1294] -0.00623688*inputs[1295] -0.00155083*inputs[1296] -0.0127092*inputs[1297] +0.0309443*inputs[1298] -0.0015508*inputs[1299] -0.0015509*inputs[1300] -0.00155082*inputs[1301] -0.0015508*inputs[1302] +0.00399118*inputs[1303] -0.0015508*inputs[1304] +0.0174406*inputs[1305] +0.0174406*inputs[1306] +0.0174406*inputs[1307] -0.00155081*inputs[1308] +0.00911222*inputs[1309] -0.00155086*inputs[1310] -0.0015508*inputs[1311] -0.00155084*inputs[1312] -0.0291185*inputs[1313] -0.0291185*inputs[1314] -0.00155085*inputs[1315] -0.00155082*inputs[1316] -0.00443253*inputs[1317] -0.00155081*inputs[1318] -0.0165648*inputs[1319] +0.0286865*inputs[1320] -0.00155082*inputs[1321] -0.00614704*inputs[1322] -0.0015508*inputs[1323] -0.0015508*inputs[1324] +0.0365684*inputs[1325] +0.00863777*inputs[1326] -0.00155084*inputs[1327] -0.0015508*inputs[1328] -0.00155082*inputs[1329] -0.0015508*inputs[1330] -0.00729889*inputs[1331] -0.0015509*inputs[1332] -0.0015508*inputs[1333] -0.00155081*inputs[1334] -0.00155085*inputs[1335] +0.0130028*inputs[1336] +0.0130028*inputs[1337] +0.0130028*inputs[1338] -0.0109409*inputs[1339] -0.0109409*inputs[1340] -0.00155085*inputs[1341] -0.00735101*inputs[1342] -0.00155081*inputs[1343] -0.00155084*inputs[1344] -0.00155081*inputs[1345] -0.0166359*inputs[1346] -0.00155085*inputs[1347] -0.00155082*inputs[1348] -0.00155091*inputs[1349] -0.00155086*inputs[1350] -0.0015508*inputs[1351] -0.00155086*inputs[1352] +0.012115*inputs[1353] +0.012115*inputs[1354] -0.00155083*inputs[1355] -0.0015508*inputs[1356] +0.0333456*inputs[1357] +0.0306794*inputs[1358] -0.00155081*inputs[1359] -0.00155081*inputs[1360] -0.00155086*inputs[1361] -0.00155083*inputs[1362] -0.00155081*inputs[1363] +0.0218794*inputs[1364] +0.0218795*inputs[1365] +0.000919273*inputs[1366] +0.000919344*inputs[1367] +0.000919238*inputs[1368] +0.00091925*inputs[1369] +0.000919328*inputs[1370] -0.0234396*inputs[1371] -0.00623432*inputs[1372] -0.00623431*inputs[1373] +0.0081797*inputs[1374] -0.00321251*inputs[1375] -0.00321251*inputs[1376] -0.0113394*inputs[1377] -0.0494524*inputs[1378] +0.00174283*inputs[1379] -0.00155081*inputs[1380] -0.00155081*inputs[1381] -0.00155085*inputs[1382] -0.00155081*inputs[1383] -0.0334978*inputs[1384] -0.0164639*inputs[1385] -0.0164639*inputs[1386] -0.00155087*inputs[1387] -0.00155081*inputs[1388] -0.0015508*inputs[1389] -0.0308487*inputs[1390] -0.00155083*inputs[1391] -0.00155082*inputs[1392] -0.00155084*inputs[1393] -0.00155082*inputs[1394] -0.00155083*inputs[1395] 
		
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

		combinations[0] = 0.0201275 +1.253*inputs[0] +1.25792*inputs[1] -1.21994*inputs[2] -1.17196*inputs[3] +1.25106*inputs[4] -1.20134*inputs[5] 
		
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
