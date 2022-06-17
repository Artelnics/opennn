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
 
		self.parameters_number = 1399
 
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

		combinations = [None] * 1

		combinations[0] = -0.0637057 -0.0750419*inputs[0] -0.208604*inputs[1] -0.447759*inputs[2] -0.204778*inputs[3] -0.0603662*inputs[4] -0.0744593*inputs[5] -0.0173452*inputs[6] -0.0533713*inputs[7] +0.018167*inputs[8] -0.000550941*inputs[9] -0.00764318*inputs[10] -0.0353391*inputs[11] -0.0140487*inputs[12] -0.107407*inputs[13] -0.108352*inputs[14] -0.107222*inputs[15] -0.0123462*inputs[16] +0.0128182*inputs[17] -0.419615*inputs[18] +0.138458*inputs[19] -0.0281388*inputs[20] -0.127183*inputs[21] +0.018394*inputs[22] -0.137731*inputs[23] +0.0339494*inputs[24] -0.02496*inputs[25] -0.179556*inputs[26] +0.0555365*inputs[27] -0.0185962*inputs[28] -0.0547835*inputs[29] +0.0495072*inputs[30] +0.110953*inputs[31] +0.201887*inputs[32] -0.24895*inputs[33] +0.182459*inputs[34] -0.220292*inputs[35] -0.186359*inputs[36] -0.00468534*inputs[37] -0.17284*inputs[38] +0.326601*inputs[39] +0.234571*inputs[40] -0.0742307*inputs[41] -0.13374*inputs[42] -0.0983385*inputs[43] -0.0938421*inputs[44] -0.0183773*inputs[45] +0.111905*inputs[46] +0.147939*inputs[47] +0.0853049*inputs[48] -0.215207*inputs[49] -0.0134395*inputs[50] +0.0292973*inputs[51] -0.0397507*inputs[52] -0.00178797*inputs[53] +0.03939*inputs[54] +0.219451*inputs[55] +0.0242822*inputs[56] +0.00217705*inputs[57] +0.315972*inputs[58] +0.115792*inputs[59] +0.273414*inputs[60] +0.117966*inputs[61] +0.0575219*inputs[62] -0.185945*inputs[63] -0.019906*inputs[64] +0.277377*inputs[65] +0.214797*inputs[66] -0.0598684*inputs[67] +0.0463823*inputs[68] +0.00290626*inputs[69] -0.151252*inputs[70] -0.00993099*inputs[71] -0.04504*inputs[72] +0.0326709*inputs[73] +0.193087*inputs[74] -0.00852797*inputs[75] -0.0939928*inputs[76] -0.0793491*inputs[77] +0.170765*inputs[78] -0.0502273*inputs[79] +0.314468*inputs[80] +0.0984663*inputs[81] -0.123422*inputs[82] -0.145103*inputs[83] +0.0803643*inputs[84] +0.065974*inputs[85] -0.12255*inputs[86] +0.0845289*inputs[87] -0.0471712*inputs[88] +0.0848125*inputs[89] -0.0211535*inputs[90] -0.060773*inputs[91] +0.0174829*inputs[92] +0.104354*inputs[93] -0.0417881*inputs[94] +0.168344*inputs[95] +0.163683*inputs[96] -0.169965*inputs[97] +0.0455784*inputs[98] -0.0263592*inputs[99] -0.0498327*inputs[100] +0.0690676*inputs[101] +0.0146016*inputs[102] +0.0242725*inputs[103] +0.0642026*inputs[104] -0.0559838*inputs[105] -0.0996685*inputs[106] +0.000645895*inputs[107] +0.117756*inputs[108] -0.0267265*inputs[109] +0.023209*inputs[110] -0.0172749*inputs[111] +0.0102732*inputs[112] -0.00592025*inputs[113] +0.182919*inputs[114] -0.00982807*inputs[115] +0.114831*inputs[116] +0.0413606*inputs[117] +0.0790646*inputs[118] +0.0250974*inputs[119] +0.0450667*inputs[120] -0.121584*inputs[121] -0.142819*inputs[122] +0.177704*inputs[123] +0.0983308*inputs[124] +0.0392111*inputs[125] -0.00843644*inputs[126] +0.0415064*inputs[127] +0.00729681*inputs[128] -0.00926891*inputs[129] +0.000431033*inputs[130] -0.0643243*inputs[131] -0.0835888*inputs[132] +0.0431031*inputs[133] +0.101453*inputs[134] +0.163319*inputs[135] +0.270176*inputs[136] -0.0352606*inputs[137] +0.148024*inputs[138] +0.18871*inputs[139] -0.0626486*inputs[140] -0.088753*inputs[141] -0.0336026*inputs[142] -0.12712*inputs[143] -0.0789421*inputs[144] -0.102179*inputs[145] +0.0285619*inputs[146] +0.0368369*inputs[147] +0.0470136*inputs[148] +0.104171*inputs[149] -0.023483*inputs[150] +0.104422*inputs[151] +0.177882*inputs[152] -0.0681189*inputs[153] +0.153907*inputs[154] +0.0140269*inputs[155] +0.0367237*inputs[156] +0.0078875*inputs[157] +0.0335578*inputs[158] -0.0770914*inputs[159] -0.0370714*inputs[160] -0.0099941*inputs[161] -0.0172684*inputs[162] +0.0472976*inputs[163] +0.0231492*inputs[164] +0.0767622*inputs[165] -0.0284412*inputs[166] -0.0430166*inputs[167] -0.152088*inputs[168] +0.0531187*inputs[169] +0.0403571*inputs[170] +0.0977621*inputs[171] -0.000161405*inputs[172] -0.0748359*inputs[173] +0.118698*inputs[174] -0.110001*inputs[175] -0.0276036*inputs[176] -0.021526*inputs[177] +0.238466*inputs[178] -0.0944925*inputs[179] +0.0937662*inputs[180] +0.0118111*inputs[181] +0.000677544*inputs[182] +0.00968449*inputs[183] +0.00494418*inputs[184] -0.0337548*inputs[185] +0.0258166*inputs[186] -0.0209334*inputs[187] -0.0316653*inputs[188] +0.0535078*inputs[189] -0.00836816*inputs[190] -0.0157062*inputs[191] -0.086513*inputs[192] +0.057261*inputs[193] -0.108443*inputs[194] -0.02402*inputs[195] -0.000283227*inputs[196] +0.0554251*inputs[197] -0.00292334*inputs[198] +0.0156918*inputs[199] +0.0128582*inputs[200] -0.030327*inputs[201] -0.0126614*inputs[202] +0.071685*inputs[203] -0.0291088*inputs[204] +0.14503*inputs[205] -0.0102659*inputs[206] +0.013659*inputs[207] +0.0538097*inputs[208] -0.00371125*inputs[209] -0.0139392*inputs[210] +0.00365399*inputs[211] +0.0211818*inputs[212] +0.0404477*inputs[213] -0.0398204*inputs[214] +0.00451095*inputs[215] +0.0274023*inputs[216] +0.0387226*inputs[217] -0.0239453*inputs[218] +0.05003*inputs[219] -0.0601483*inputs[220] +0.0448513*inputs[221] -0.0844066*inputs[222] -0.0102479*inputs[223] -0.0204018*inputs[224] +0.0385134*inputs[225] -0.0277163*inputs[226] +0.118181*inputs[227] -0.0284621*inputs[228] +0.08601*inputs[229] -0.231525*inputs[230] -0.0548602*inputs[231] -0.0369787*inputs[232] -0.0315901*inputs[233] +0.0106185*inputs[234] -0.0732641*inputs[235] +0.0479642*inputs[236] -0.0569726*inputs[237] +0.0347851*inputs[238] +0.111331*inputs[239] +0.0415826*inputs[240] +0.0243007*inputs[241] -0.013992*inputs[242] -0.0536448*inputs[243] -0.0308712*inputs[244] -0.135017*inputs[245] -0.00842317*inputs[246] -0.00283517*inputs[247] +0.107792*inputs[248] +0.00678911*inputs[249] -0.0261884*inputs[250] +0.0264577*inputs[251] +0.086788*inputs[252] +0.104558*inputs[253] +0.00597823*inputs[254] +0.0470159*inputs[255] -0.0019317*inputs[256] +0.0228791*inputs[257] +0.0129491*inputs[258] +0.0484728*inputs[259] +0.0342285*inputs[260] +0.00300293*inputs[261] -0.00837746*inputs[262] +0.028447*inputs[263] +0.000414568*inputs[264] +0.0374702*inputs[265] -0.100288*inputs[266] +0.040322*inputs[267] +0.0617664*inputs[268] -0.0238129*inputs[269] +0.00416179*inputs[270] +0.122867*inputs[271] -0.0132633*inputs[272] -0.119503*inputs[273] -0.0775472*inputs[274] +0.0328964*inputs[275] -0.0538451*inputs[276] -0.00856023*inputs[277] -0.0140114*inputs[278] -0.0608802*inputs[279] -0.0465888*inputs[280] +0.0112457*inputs[281] -0.0203884*inputs[282] -0.0466154*inputs[283] +0.11476*inputs[284] -0.0853528*inputs[285] +0.0481008*inputs[286] +0.0319512*inputs[287] +0.00403464*inputs[288] -0.034222*inputs[289] -0.0693127*inputs[290] +0.027726*inputs[291] +0.00707507*inputs[292] +0.0661649*inputs[293] -0.0515176*inputs[294] +0.031563*inputs[295] +0.00554168*inputs[296] -0.0187324*inputs[297] -0.0510183*inputs[298] -0.130714*inputs[299] -0.0854016*inputs[300] -0.0376345*inputs[301] -0.0234266*inputs[302] +0.123669*inputs[303] +0.00496026*inputs[304] +0.0354245*inputs[305] +0.0315233*inputs[306] +0.00373112*inputs[307] +0.0370584*inputs[308] +0.0349722*inputs[309] -0.0492234*inputs[310] +0.00503345*inputs[311] +0.164518*inputs[312] -0.051573*inputs[313] -0.0217118*inputs[314] +0.00403452*inputs[315] +7.20074e-05*inputs[316] +0.00407098*inputs[317] -0.0578196*inputs[318] -0.0567176*inputs[319] +0.124618*inputs[320] +0.039009*inputs[321] +0.00349061*inputs[322] +0.011761*inputs[323] +0.0196611*inputs[324] -0.00429477*inputs[325] +0.00799223*inputs[326] -0.0457582*inputs[327] -0.0138565*inputs[328] -0.029108*inputs[329] -0.0300801*inputs[330] +0.0280831*inputs[331] -0.0333558*inputs[332] -0.00734257*inputs[333] +0.0233536*inputs[334] +0.0269435*inputs[335] -0.00451039*inputs[336] -0.0729836*inputs[337] +0.123265*inputs[338] +0.00217416*inputs[339] +0.0367883*inputs[340] -0.106388*inputs[341] +0.0301074*inputs[342] -0.0467761*inputs[343] +0.00349219*inputs[344] -0.0603668*inputs[345] +0.0789268*inputs[346] +0.0125188*inputs[347] +0.0641917*inputs[348] -0.0130456*inputs[349] +0.0285247*inputs[350] +0.0171203*inputs[351] +0.0317679*inputs[352] +0.00349016*inputs[353] -0.0105164*inputs[354] -0.0153876*inputs[355] -0.00260174*inputs[356] -0.00110007*inputs[357] -0.0783857*inputs[358] -0.120425*inputs[359] -0.019501*inputs[360] +0.0518125*inputs[361] +0.0133961*inputs[362] +0.0283802*inputs[363] -0.000467765*inputs[364] +0.0337926*inputs[365] +0.0034906*inputs[366] -0.00651654*inputs[367] -0.0655399*inputs[368] +0.0126714*inputs[369] -0.000925885*inputs[370] +0.0578867*inputs[371] +0.0291636*inputs[372] +0.0366892*inputs[373] +0.0232578*inputs[374] +0.0320618*inputs[375] -1.02827e-05*inputs[376] -0.0363734*inputs[377] -0.01983*inputs[378] -0.0621186*inputs[379] +0.0257227*inputs[380] -0.0978381*inputs[381] -0.00584654*inputs[382] -0.00209649*inputs[383] +0.00349219*inputs[384] -0.0152201*inputs[385] +0.0315384*inputs[386] +0.00708025*inputs[387] +0.00368544*inputs[388] +0.00983195*inputs[389] +0.036932*inputs[390] +0.00349099*inputs[391] +0.00333455*inputs[392] -0.076834*inputs[393] +0.0252398*inputs[394] -0.0382849*inputs[395] +0.0347806*inputs[396] -0.0439987*inputs[397] -0.015741*inputs[398] +0.00736705*inputs[399] -0.118296*inputs[400] +0.0313752*inputs[401] +0.0329435*inputs[402] -0.0313667*inputs[403] +0.0273223*inputs[404] +0.0125091*inputs[405] +0.119629*inputs[406] +0.0391402*inputs[407] +0.0395502*inputs[408] +0.0291884*inputs[409] -0.0185651*inputs[410] +0.124104*inputs[411] -0.0618643*inputs[412] +0.0256015*inputs[413] +0.0255781*inputs[414] +0.0413357*inputs[415] +0.00937894*inputs[416] +0.023335*inputs[417] +0.00349239*inputs[418] +0.045156*inputs[419] -0.0153127*inputs[420] +0.125291*inputs[421] +0.00216869*inputs[422] +0.0574781*inputs[423] +0.0582644*inputs[424] -0.0557453*inputs[425] -0.00148468*inputs[426] -0.037738*inputs[427] -0.018086*inputs[428] -0.0228471*inputs[429] +0.00349183*inputs[430] +0.0732505*inputs[431] -0.00060189*inputs[432] -0.0315399*inputs[433] +0.0407041*inputs[434] +0.11961*inputs[435] +0.129516*inputs[436] +0.00349013*inputs[437] +0.0397547*inputs[438] +0.0553245*inputs[439] +0.041302*inputs[440] +0.10781*inputs[441] +0.00349016*inputs[442] -0.0169381*inputs[443] +0.0190463*inputs[444] +0.0459544*inputs[445] +0.130396*inputs[446] -0.0601558*inputs[447] +0.0205258*inputs[448] -0.0138535*inputs[449] +0.0359606*inputs[450] +0.0935517*inputs[451] +0.023611*inputs[452] +0.023017*inputs[453] -0.0139198*inputs[454] +0.073995*inputs[455] +0.073995*inputs[456] +0.00284916*inputs[457] -0.0285949*inputs[458] +0.00284864*inputs[459] -0.0207807*inputs[460] +0.0270745*inputs[461] +0.0209404*inputs[462] +0.0561537*inputs[463] +0.0375591*inputs[464] +0.0553615*inputs[465] +0.00863528*inputs[466] +0.0370832*inputs[467] -0.0207826*inputs[468] +0.00138629*inputs[469] +0.0176441*inputs[470] -0.0420668*inputs[471] -0.10891*inputs[472] +0.0337667*inputs[473] -0.0185896*inputs[474] +0.0391925*inputs[475] +0.0806354*inputs[476] +0.0495066*inputs[477] +0.00284815*inputs[478] +0.0460651*inputs[479] +0.064888*inputs[480] -0.0474724*inputs[481] +0.0567973*inputs[482] +0.0280454*inputs[483] +0.0876056*inputs[484] +0.00434046*inputs[485] +0.0218869*inputs[486] +0.00284896*inputs[487] -0.061177*inputs[488] +0.0390954*inputs[489] +0.134707*inputs[490] -0.0184852*inputs[491] +0.0103617*inputs[492] +0.053842*inputs[493] -0.0339911*inputs[494] -0.00561933*inputs[495] +0.00284944*inputs[496] +0.0947363*inputs[497] +0.0468959*inputs[498] -0.0428934*inputs[499] -0.0886445*inputs[500] +0.0127231*inputs[501] +0.0113074*inputs[502] -0.0119752*inputs[503] +0.0363989*inputs[504] +0.00285008*inputs[505] -0.00460975*inputs[506] -0.0344875*inputs[507] -0.0120614*inputs[508] +0.00284798*inputs[509] -0.0272397*inputs[510] -0.0242086*inputs[511] +0.0713262*inputs[512] +0.00284798*inputs[513] +0.0416779*inputs[514] +0.122269*inputs[515] +0.0149558*inputs[516] +0.0180329*inputs[517] +0.0114362*inputs[518] +0.0293391*inputs[519] +0.0401453*inputs[520] +0.042205*inputs[521] +0.00284994*inputs[522] +0.00284931*inputs[523] +0.0455676*inputs[524] +0.0028482*inputs[525] +0.00321094*inputs[526] +0.0468963*inputs[527] -0.0325065*inputs[528] +0.0106105*inputs[529] +0.0271471*inputs[530] -0.122062*inputs[531] -0.0147634*inputs[532] +0.0294712*inputs[533] -0.0552851*inputs[534] -0.0198478*inputs[535] +0.0412795*inputs[536] +0.00284987*inputs[537] +0.00284797*inputs[538] -0.0272192*inputs[539] -0.0272192*inputs[540] +0.0169159*inputs[541] -0.018238*inputs[542] -0.00832804*inputs[543] +0.0256608*inputs[544] +0.00284896*inputs[545] +0.01007*inputs[546] -0.00768492*inputs[547] +0.0401452*inputs[548] +0.0570962*inputs[549] -0.0420665*inputs[550] +0.00284799*inputs[551] +0.0604394*inputs[552] -0.0198478*inputs[553] -0.0136005*inputs[554] +0.00284885*inputs[555] -0.0227537*inputs[556] +0.0729224*inputs[557] +0.0180328*inputs[558] +0.130244*inputs[559] +0.00400195*inputs[560] -0.110232*inputs[561] +0.00284927*inputs[562] +0.047838*inputs[563] +0.00285001*inputs[564] +0.00285007*inputs[565] +0.0137724*inputs[566] +0.0803561*inputs[567] +0.0788887*inputs[568] -0.0442237*inputs[569] -0.0223066*inputs[570] +0.036399*inputs[571] +0.0239923*inputs[572] -0.0743904*inputs[573] -0.0203803*inputs[574] +0.00275208*inputs[575] -0.00593363*inputs[576] +0.00275068*inputs[577] +0.00308643*inputs[578] +0.0495439*inputs[579] -0.0195415*inputs[580] +0.0129835*inputs[581] +0.0114366*inputs[582] +0.041737*inputs[583] -0.00132907*inputs[584] +0.055556*inputs[585] -0.0932026*inputs[586] +0.0261155*inputs[587] -0.0501202*inputs[588] -0.0082261*inputs[589] +0.0396108*inputs[590] -0.000791408*inputs[591] +0.0600221*inputs[592] +0.0306018*inputs[593] -0.042333*inputs[594] -0.00698729*inputs[595] -0.0626671*inputs[596] +0.06063*inputs[597] +0.00284964*inputs[598] -0.0317327*inputs[599] +0.0191343*inputs[600] +0.106156*inputs[601] -0.0211262*inputs[602] -0.0515891*inputs[603] -0.049186*inputs[604] +0.00434054*inputs[605] +0.00284853*inputs[606] +0.0028491*inputs[607] +0.0251577*inputs[608] +0.00284799*inputs[609] -0.0394782*inputs[610] -0.0109841*inputs[611] +0.0337985*inputs[612] -0.00698976*inputs[613] +0.0297737*inputs[614] +0.0185279*inputs[615] +0.00284993*inputs[616] +0.0230085*inputs[617] +0.0137723*inputs[618] +0.0165212*inputs[619] +0.00327818*inputs[620] +0.00284969*inputs[621] +0.040412*inputs[622] +0.00285005*inputs[623] +0.00284968*inputs[624] +0.00284996*inputs[625] +0.00523573*inputs[626] +0.00284975*inputs[627] -0.0293234*inputs[628] +0.0161345*inputs[629] +0.0279854*inputs[630] -0.0533924*inputs[631] +4.7885e-06*inputs[632] -0.0146841*inputs[633] -0.00668247*inputs[634] +0.000979908*inputs[635] -0.00668222*inputs[636] +0.00285014*inputs[637] -0.0686144*inputs[638] -0.0338863*inputs[639] -0.0572545*inputs[640] +0.0163577*inputs[641] +0.0135262*inputs[642] +0.0028503*inputs[643] -0.00397629*inputs[644] +0.00886382*inputs[645] +0.024576*inputs[646] +0.00285029*inputs[647] +0.0142834*inputs[648] +0.0114567*inputs[649] +0.0198199*inputs[650] +0.0305203*inputs[651] +0.0305207*inputs[652] +0.03052*inputs[653] +0.0020136*inputs[654] +0.00201248*inputs[655] +0.0171254*inputs[656] +0.0171255*inputs[657] +0.0171253*inputs[658] +0.0171253*inputs[659] +0.0171253*inputs[660] +0.00201364*inputs[661] +0.0504042*inputs[662] +0.0229258*inputs[663] +0.022926*inputs[664] +0.0506474*inputs[665] -0.0143823*inputs[666] +0.00201401*inputs[667] +0.00201387*inputs[668] +0.085355*inputs[669] -0.0312655*inputs[670] +0.04102*inputs[671] +0.00201431*inputs[672] +0.00201418*inputs[673] +0.00201271*inputs[674] +0.00201325*inputs[675] +0.0289232*inputs[676] +0.0289233*inputs[677] +0.0289247*inputs[678] +0.00201393*inputs[679] -0.0418896*inputs[680] -0.0418894*inputs[681] -0.0418894*inputs[682] +0.00201271*inputs[683] +0.00201411*inputs[684] +0.00135017*inputs[685] +0.0013498*inputs[686] +0.0906893*inputs[687] +0.00201432*inputs[688] +0.00201363*inputs[689] +0.0391476*inputs[690] +0.00201362*inputs[691] -0.00853205*inputs[692] -0.00853201*inputs[693] -0.00853009*inputs[694] -0.00853065*inputs[695] -0.0238878*inputs[696] -0.0238878*inputs[697] +0.00201346*inputs[698] +0.00201322*inputs[699] +0.00201247*inputs[700] +0.00201247*inputs[701] +0.00201394*inputs[702] +0.00201397*inputs[703] +0.0020137*inputs[704] +0.00201405*inputs[705] -0.0150304*inputs[706] +0.0020143*inputs[707] +0.00201412*inputs[708] -0.0481507*inputs[709] +0.00201363*inputs[710] +0.000549691*inputs[711] +0.00201392*inputs[712] +0.00201317*inputs[713] +0.00201303*inputs[714] +0.0869057*inputs[715] +0.00201399*inputs[716] +0.00201327*inputs[717] +0.00201414*inputs[718] +0.00201296*inputs[719] +0.0304556*inputs[720] +0.0304549*inputs[721] +0.00201398*inputs[722] +0.0386723*inputs[723] +0.0020125*inputs[724] +0.00201434*inputs[725] +0.00201257*inputs[726] +0.00201249*inputs[727] +0.00201363*inputs[728] +0.00201251*inputs[729] -0.00593177*inputs[730] -0.00593218*inputs[731] -0.00593289*inputs[732] -0.00593282*inputs[733] -0.00593219*inputs[734] -0.0153041*inputs[735] -0.0153044*inputs[736] -0.0740303*inputs[737] -0.0432615*inputs[738] -0.0691127*inputs[739] +0.0727958*inputs[740] -0.0710784*inputs[741] +0.00201395*inputs[742] +0.00201377*inputs[743] +0.00141027*inputs[744] -0.0770221*inputs[745] -0.022871*inputs[746] -0.0228694*inputs[747] +0.0250357*inputs[748] +0.0250339*inputs[749] +0.00201369*inputs[750] +0.0020127*inputs[751] -0.0175406*inputs[752] +0.00201352*inputs[753] +0.00201394*inputs[754] +0.00201327*inputs[755] +0.0020141*inputs[756] +0.00201429*inputs[757] +0.0020141*inputs[758] +0.00201392*inputs[759] +0.00201408*inputs[760] +0.0804819*inputs[761] +0.00201429*inputs[762] +0.00201329*inputs[763] +0.00201375*inputs[764] +0.00201405*inputs[765] +0.00201423*inputs[766] +0.00201422*inputs[767] +0.0297099*inputs[768] +0.02971*inputs[769] +0.0700814*inputs[770] -0.0263154*inputs[771] -0.0263155*inputs[772] +0.00201433*inputs[773] +0.00201247*inputs[774] +0.0163149*inputs[775] +0.0163164*inputs[776] +0.0163149*inputs[777] +0.0163159*inputs[778] +0.0163171*inputs[779] +0.00201267*inputs[780] +0.0787075*inputs[781] +0.00201312*inputs[782] +0.00201283*inputs[783] +0.0296849*inputs[784] +0.0296842*inputs[785] +0.0401372*inputs[786] +0.00916683*inputs[787] +0.00916761*inputs[788] +0.00916567*inputs[789] -0.0222256*inputs[790] +0.0396081*inputs[791] +0.00201431*inputs[792] +0.0396443*inputs[793] +0.0396454*inputs[794] +0.0020142*inputs[795] +0.00201248*inputs[796] -0.062398*inputs[797] +0.0122042*inputs[798] +0.0122031*inputs[799] +0.0122037*inputs[800] +0.00201385*inputs[801] +0.00201415*inputs[802] +0.0346793*inputs[803] +0.0188666*inputs[804] +0.0245833*inputs[805] +0.00201422*inputs[806] -0.0297254*inputs[807] +0.00201422*inputs[808] +0.00201435*inputs[809] +0.00201315*inputs[810] +0.00201385*inputs[811] +0.0245832*inputs[812] -0.0577576*inputs[813] +0.00201419*inputs[814] +0.00201291*inputs[815] +0.0409467*inputs[816] -0.0297253*inputs[817] +0.00201423*inputs[818] +0.00201247*inputs[819] +0.00211436*inputs[820] +0.00201348*inputs[821] +0.0822584*inputs[822] +0.0704387*inputs[823] +0.00201386*inputs[824] +0.00201387*inputs[825] +0.00201432*inputs[826] -0.0297253*inputs[827] +0.00201387*inputs[828] -0.028884*inputs[829] -0.0300697*inputs[830] -0.0328916*inputs[831] -0.0328913*inputs[832] +0.0020143*inputs[833] +0.00201381*inputs[834] -0.0328914*inputs[835] -0.0308224*inputs[836] -0.0480622*inputs[837] +0.0020137*inputs[838] -0.039381*inputs[839] -0.0576628*inputs[840] +0.00201416*inputs[841] -0.059723*inputs[842] +0.0101918*inputs[843] +0.0101917*inputs[844] +0.0101918*inputs[845] +0.0101918*inputs[846] +0.0101918*inputs[847] -0.0353239*inputs[848] +0.0718787*inputs[849] +0.0718789*inputs[850] -0.0353242*inputs[851] +0.0406776*inputs[852] +0.0253983*inputs[853] -0.0669944*inputs[854] -0.0156452*inputs[855] +0.00201251*inputs[856] +0.00201271*inputs[857] +0.0465692*inputs[858] -0.0318189*inputs[859] +0.0644031*inputs[860] +0.0278679*inputs[861] +0.0162256*inputs[862] +0.0162274*inputs[863] +0.0162269*inputs[864] -0.0350992*inputs[865] +0.00201303*inputs[866] +0.025399*inputs[867] +0.0453283*inputs[868] -0.0759489*inputs[869] +0.0276191*inputs[870] +0.0276195*inputs[871] +0.013406*inputs[872] +0.0191258*inputs[873] +0.0191259*inputs[874] +0.0191258*inputs[875] +0.00183127*inputs[876] -0.0308224*inputs[877] +0.043271*inputs[878] +0.0521*inputs[879] +0.0432708*inputs[880] -0.0135367*inputs[881] +0.0871339*inputs[882] +0.0151714*inputs[883] +0.0151716*inputs[884] +0.002014*inputs[885] +0.00201382*inputs[886] +0.00201412*inputs[887] +0.0066334*inputs[888] +0.00663251*inputs[889] -0.0684661*inputs[890] +0.0020142*inputs[891] +0.00201415*inputs[892] +0.00201374*inputs[893] +0.00201369*inputs[894] +0.00201388*inputs[895] +0.00201415*inputs[896] +0.0130046*inputs[897] +0.00201435*inputs[898] +0.00201467*inputs[899] +0.0130046*inputs[900] -0.0308224*inputs[901] -0.00995165*inputs[902] +0.0454294*inputs[903] -0.0280057*inputs[904] -0.0099515*inputs[905] -0.00995131*inputs[906] -0.00108145*inputs[907] -0.00108151*inputs[908] +0.00201404*inputs[909] +0.00201385*inputs[910] +0.0122196*inputs[911] +0.00201429*inputs[912] +0.00201395*inputs[913] -0.0167846*inputs[914] -0.0167845*inputs[915] +0.0454305*inputs[916] +0.00201375*inputs[917] -0.0377696*inputs[918] +0.00201299*inputs[919] +0.0020125*inputs[920] +0.0454293*inputs[921] +0.0706808*inputs[922] +0.00201425*inputs[923] +0.00201396*inputs[924] +0.0623948*inputs[925] +0.00201434*inputs[926] +0.00201403*inputs[927] -0.0377715*inputs[928] +0.105693*inputs[929] -0.0385654*inputs[930] -0.0385649*inputs[931] +0.00201279*inputs[932] +0.0280456*inputs[933] -0.0175955*inputs[934] +0.0129837*inputs[935] +0.0280454*inputs[936] +0.00201377*inputs[937] -0.0175954*inputs[938] +0.00201339*inputs[939] +0.0171831*inputs[940] +0.0171827*inputs[941] +0.0171839*inputs[942] +0.0171823*inputs[943] +0.0020134*inputs[944] +0.0458122*inputs[945] +0.00201276*inputs[946] -0.0216906*inputs[947] +0.0634479*inputs[948] -0.00815242*inputs[949] -0.00815242*inputs[950] +0.0458141*inputs[951] +0.0921659*inputs[952] +0.0803791*inputs[953] +0.00201412*inputs[954] +0.00201406*inputs[955] +0.00201419*inputs[956] +0.00201362*inputs[957] +0.00201248*inputs[958] +0.0363488*inputs[959] +0.00201396*inputs[960] +0.00201331*inputs[961] +0.000639263*inputs[962] +0.00063928*inputs[963] +0.00201388*inputs[964] -0.0279001*inputs[965] +0.00201248*inputs[966] +0.00201247*inputs[967] +0.00201306*inputs[968] +0.0822606*inputs[969] -0.0693295*inputs[970] +0.0020142*inputs[971] +0.00234811*inputs[972] +0.00234802*inputs[973] +0.00234811*inputs[974] +0.00234813*inputs[975] +0.00201285*inputs[976] +0.0539774*inputs[977] +0.00201288*inputs[978] +0.063698*inputs[979] +0.00201393*inputs[980] -0.0698498*inputs[981] +0.00201376*inputs[982] +0.00201248*inputs[983] +0.00201369*inputs[984] +0.00201408*inputs[985] +0.0179448*inputs[986] +0.00201339*inputs[987] +0.00201325*inputs[988] +0.0412426*inputs[989] +0.00201248*inputs[990] +0.00906068*inputs[991] +0.00201356*inputs[992] +0.0234752*inputs[993] -4.56869e-05*inputs[994] +0.0234753*inputs[995] +0.0506433*inputs[996] +0.00201288*inputs[997] +0.00906182*inputs[998] +0.00234302*inputs[999] +0.00234301*inputs[1000] +0.00234303*inputs[1001] +0.00201403*inputs[1002] +0.00201405*inputs[1003] +0.00906019*inputs[1004] -0.0382596*inputs[1005] +0.0254764*inputs[1006] +0.0254764*inputs[1007] +0.00201273*inputs[1008] -0.0333515*inputs[1009] -0.0333515*inputs[1010] -0.0333514*inputs[1011] +0.0163762*inputs[1012] +0.0163763*inputs[1013] +0.0822593*inputs[1014] -0.0308775*inputs[1015] +0.0221505*inputs[1016] +0.022149*inputs[1017] -0.0468687*inputs[1018] -0.0468692*inputs[1019] +0.064274*inputs[1020] +0.00201248*inputs[1021] +0.0260914*inputs[1022] +0.00201262*inputs[1023] +0.00906015*inputs[1024] +0.00453951*inputs[1025] +0.00453951*inputs[1026] +0.00453953*inputs[1027] +0.0456236*inputs[1028] +0.0456237*inputs[1029] +0.0456237*inputs[1030] +0.00906176*inputs[1031] +0.00201368*inputs[1032] -0.0339653*inputs[1033] +0.00201419*inputs[1034] +0.00201247*inputs[1035] +0.00201408*inputs[1036] +0.00201418*inputs[1037] +0.00201248*inputs[1038] +0.00201368*inputs[1039] +0.0102634*inputs[1040] +0.0260909*inputs[1041] -0.0318769*inputs[1042] -0.024265*inputs[1043] +0.0526184*inputs[1044] +0.0526183*inputs[1045] +0.0526182*inputs[1046] -0.0242654*inputs[1047] -0.0228577*inputs[1048] +0.00201255*inputs[1049] +0.00445702*inputs[1050] -0.0228575*inputs[1051] -0.0242652*inputs[1052] +0.00201385*inputs[1053] -0.0118941*inputs[1054] -0.0118941*inputs[1055] -0.0118963*inputs[1056] -0.0118939*inputs[1057] +0.0394573*inputs[1058] +0.00201381*inputs[1059] +0.0625883*inputs[1060] -0.00731492*inputs[1061] -0.00731484*inputs[1062] -0.0110765*inputs[1063] -0.0740309*inputs[1064] -0.0149459*inputs[1065] +0.00201417*inputs[1066] +0.00201304*inputs[1067] -0.00146001*inputs[1068] -0.00146157*inputs[1069] -0.00146065*inputs[1070] +0.0261355*inputs[1071] +0.0261353*inputs[1072] -0.0149458*inputs[1073] +0.0400716*inputs[1074] -0.0149458*inputs[1075] +0.0400714*inputs[1076] +0.0241762*inputs[1077] +0.00201392*inputs[1078] +0.00201368*inputs[1079] +0.00201346*inputs[1080] +0.00201428*inputs[1081] +0.00201247*inputs[1082] +0.017453*inputs[1083] +0.017453*inputs[1084] +0.017453*inputs[1085] +0.00261921*inputs[1086] +0.00261921*inputs[1087] +0.00201274*inputs[1088] +0.0679649*inputs[1089] +0.0858847*inputs[1090] -0.0228572*inputs[1091] +0.00201425*inputs[1092] +0.00201422*inputs[1093] +0.0020125*inputs[1094] +0.00538617*inputs[1095] -0.0228577*inputs[1096] +0.00538617*inputs[1097] +0.00538621*inputs[1098] +0.00201376*inputs[1099] +0.00201403*inputs[1100] +0.00202526*inputs[1101] +0.00202519*inputs[1102] +0.00201252*inputs[1103] +0.00201364*inputs[1104] +0.00201394*inputs[1105] +0.00201427*inputs[1106] +0.00201275*inputs[1107] +0.00201257*inputs[1108] +0.00201251*inputs[1109] -0.0137843*inputs[1110] -0.0647437*inputs[1111] +0.002013*inputs[1112] +0.00201432*inputs[1113] +0.00788708*inputs[1114] -0.0298117*inputs[1115] +0.0496982*inputs[1116] +0.00201416*inputs[1117] +0.00201253*inputs[1118] -0.0614596*inputs[1119] -0.0149459*inputs[1120] +0.0750558*inputs[1121] +0.00201323*inputs[1122] +0.00201323*inputs[1123] +0.00201248*inputs[1124] +0.0746096*inputs[1125] +0.0375432*inputs[1126] +0.037545*inputs[1127] +0.00777656*inputs[1128] +0.00777659*inputs[1129] +0.0628013*inputs[1130] +0.049436*inputs[1131] -0.0983507*inputs[1132] +0.00201341*inputs[1133] +0.00201349*inputs[1134] +0.03174*inputs[1135] +0.03174*inputs[1136] +0.0317402*inputs[1137] +0.0436064*inputs[1138] +0.00201247*inputs[1139] +0.00201416*inputs[1140] -0.0500617*inputs[1141] +0.00788708*inputs[1142] +0.0141514*inputs[1143] +0.0141514*inputs[1144] +0.0141513*inputs[1145] +0.0141517*inputs[1146] -0.0268911*inputs[1147] -0.0268901*inputs[1148] +0.00201421*inputs[1149] +0.00201397*inputs[1150] +0.00201424*inputs[1151] +0.00201337*inputs[1152] -0.017209*inputs[1153] -0.0172091*inputs[1154] -0.0172093*inputs[1155] +0.0020139*inputs[1156] +0.00201415*inputs[1157] +0.0383505*inputs[1158] -0.0588067*inputs[1159] +0.00201295*inputs[1160] +0.00201293*inputs[1161] +0.00201247*inputs[1162] +0.00201366*inputs[1163] +0.00201262*inputs[1164] +0.00201346*inputs[1165] +0.00201398*inputs[1166] +0.00201298*inputs[1167] +0.00201427*inputs[1168] +0.0299932*inputs[1169] +0.0299932*inputs[1170] +0.0299937*inputs[1171] +0.0299962*inputs[1172] -0.0114596*inputs[1173] -0.032848*inputs[1174] -0.0328479*inputs[1175] -0.0328486*inputs[1176] -0.0328479*inputs[1177] -0.0258262*inputs[1178] -0.0258262*inputs[1179] -0.0258261*inputs[1180] -0.0258263*inputs[1181] -0.0306547*inputs[1182] -0.0306546*inputs[1183] +0.00201377*inputs[1184] +0.00201279*inputs[1185] +0.00201383*inputs[1186] +0.00201247*inputs[1187] -0.0100905*inputs[1188] -0.0100895*inputs[1189] -0.0625858*inputs[1190] -0.0100893*inputs[1191] +0.00201248*inputs[1192] +0.044417*inputs[1193] +0.0020141*inputs[1194] +0.00201309*inputs[1195] +0.00201411*inputs[1196] +0.00201267*inputs[1197] -0.0499125*inputs[1198] -0.0499128*inputs[1199] +0.0623975*inputs[1200] +0.0822509*inputs[1201] +0.00201361*inputs[1202] +0.00201415*inputs[1203] +0.0020129*inputs[1204] +0.0563353*inputs[1205] -0.00477662*inputs[1206] -0.00477662*inputs[1207] +0.00201311*inputs[1208] +0.00201356*inputs[1209] +0.00201412*inputs[1210] +0.00201256*inputs[1211] +0.00201247*inputs[1212] +0.00201433*inputs[1213] +0.00201254*inputs[1214] +0.0020143*inputs[1215] +0.0020132*inputs[1216] +0.00201386*inputs[1217] -0.0262525*inputs[1218] +0.00201293*inputs[1219] +0.00201432*inputs[1220] +0.0020127*inputs[1221] +0.0310501*inputs[1222] +0.0310518*inputs[1223] +0.0310515*inputs[1224] +0.00201435*inputs[1225] +0.00201336*inputs[1226] -0.0136418*inputs[1227] -0.0136436*inputs[1228] -0.0136418*inputs[1229] +0.0335452*inputs[1230] +0.0335473*inputs[1231] +0.002014*inputs[1232] -0.016402*inputs[1233] -0.016402*inputs[1234] -0.016402*inputs[1235] +0.00201387*inputs[1236] +0.00201379*inputs[1237] +0.054733*inputs[1238] +0.00201425*inputs[1239] +0.00201323*inputs[1240] +0.0585162*inputs[1241] +0.00589859*inputs[1242] +0.0020125*inputs[1243] +0.00589921*inputs[1244] +0.00589831*inputs[1245] +0.00201338*inputs[1246] +0.00201249*inputs[1247] +0.00201417*inputs[1248] +0.00201421*inputs[1249] -0.0263717*inputs[1250] -0.0263718*inputs[1251] +0.0259607*inputs[1252] +0.00201406*inputs[1253] +0.00201314*inputs[1254] +0.00201425*inputs[1255] +0.00201432*inputs[1256] +0.00201304*inputs[1257] +0.028929*inputs[1258] +0.0289302*inputs[1259] +0.00201366*inputs[1260] -0.00156624*inputs[1261] +0.00201296*inputs[1262] +0.00201431*inputs[1263] +0.00201416*inputs[1264] +0.00201389*inputs[1265] +0.00201431*inputs[1266] +0.0707657*inputs[1267] +0.0707654*inputs[1268] -0.0411188*inputs[1269] -0.0484722*inputs[1270] -0.116996*inputs[1271] +0.00201418*inputs[1272] +0.00201383*inputs[1273] +0.00201431*inputs[1274] +0.00201254*inputs[1275] +0.00201383*inputs[1276] +0.00201384*inputs[1277] +0.00201431*inputs[1278] +0.0342583*inputs[1279] -0.0215969*inputs[1280] -0.00156619*inputs[1281] -0.0215969*inputs[1282] +0.0600145*inputs[1283] +0.0439252*inputs[1284] +0.0347362*inputs[1285] +0.102178*inputs[1286] +0.0329451*inputs[1287] +0.0329439*inputs[1288] +0.00201281*inputs[1289] +0.00402054*inputs[1290] +0.0360908*inputs[1291] +0.0288386*inputs[1292] +0.0708028*inputs[1293] +0.00201384*inputs[1294] +0.024244*inputs[1295] +0.0866473*inputs[1296] +0.0360909*inputs[1297] -0.049939*inputs[1298] +0.00201417*inputs[1299] +0.0306003*inputs[1300] +0.0020127*inputs[1301] +0.00201379*inputs[1302] -0.0285943*inputs[1303] +0.00201274*inputs[1304] +0.00201397*inputs[1305] +0.00201344*inputs[1306] +0.00201247*inputs[1307] -0.010886*inputs[1308] -0.00995758*inputs[1309] +0.00201268*inputs[1310] +0.0020125*inputs[1311] +0.00201322*inputs[1312] +0.00201428*inputs[1313] +0.00201429*inputs[1314] +0.0343048*inputs[1315] +0.00201428*inputs[1316] +0.00201364*inputs[1317] +0.0020135*inputs[1318] +0.0565886*inputs[1319] -0.0637446*inputs[1320] +0.00201299*inputs[1321] +0.00201426*inputs[1322] -0.014145*inputs[1323] -0.0141444*inputs[1324] -0.0487022*inputs[1325] -0.0114134*inputs[1326] +0.00201271*inputs[1327] +0.0816729*inputs[1328] +0.0561544*inputs[1329] +0.0561535*inputs[1330] +0.00201422*inputs[1331] +0.00201454*inputs[1332] +0.00201458*inputs[1333] +0.00201452*inputs[1334] +0.00201453*inputs[1335] +0.00201347*inputs[1336] +0.00201429*inputs[1337] +0.00201362*inputs[1338] +0.0152131*inputs[1339] +0.0152131*inputs[1340] +0.0188665*inputs[1341] +0.0553188*inputs[1342] +0.0058256*inputs[1343] +0.00582336*inputs[1344] +0.00582433*inputs[1345] +0.0506986*inputs[1346] -0.0385886*inputs[1347] -0.0385874*inputs[1348] -0.0313899*inputs[1349] +0.00201377*inputs[1350] +0.063484*inputs[1351] +0.00201284*inputs[1352] +0.00201431*inputs[1353] +0.00201381*inputs[1354] +0.00201298*inputs[1355] -0.0739607*inputs[1356] -0.0740311*inputs[1357] +0.00201392*inputs[1358] +0.0396073*inputs[1359] -0.061474*inputs[1360] -0.00757106*inputs[1361] -0.0281218*inputs[1362] -0.028122*inputs[1363] +0.00201393*inputs[1364] +0.00201399*inputs[1365] -5.30089e-05*inputs[1366] -5.12341e-05*inputs[1367] -5.32112e-05*inputs[1368] -5.33621e-05*inputs[1369] -5.12973e-05*inputs[1370] +0.0477714*inputs[1371] +0.0139686*inputs[1372] +0.0139686*inputs[1373] -0.00933508*inputs[1374] +0.00822215*inputs[1375] +0.00822212*inputs[1376] +0.0582498*inputs[1377] +0.0892715*inputs[1378] +0.00201416*inputs[1379] +0.00201383*inputs[1380] +0.00201427*inputs[1381] +0.00201423*inputs[1382] +0.00201248*inputs[1383] +0.00201422*inputs[1384] +0.0647028*inputs[1385] +0.0647027*inputs[1386] +0.00201428*inputs[1387] +0.0576064*inputs[1388] +0.0576066*inputs[1389] +0.00201255*inputs[1390] +0.0607047*inputs[1391] +0.00201413*inputs[1392] +0.0020142*inputs[1393] +0.00201305*inputs[1394] +0.00201404*inputs[1395] 
		
		activations = [None] * 1

		activations[0] = np.tanh(combinations[0])

		return activations;


	def probabilistic_layer(self, inputs):

		combinations = [None] * 1

		combinations[0] = 0.0244542 +3.07913*inputs[0] 
		
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
