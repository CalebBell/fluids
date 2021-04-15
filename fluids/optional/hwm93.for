      SUBROUTINE GWS5(IYD,SEC,ALT,GLAT,GLONG,STL,F107A,F107,AP,W)
C      Horizontal wind model HWM93 covering all altitude regions
C      A. E. HEDIN  (1/25/93) (4/9/93)
C      Calling argument list made similar to GTS5 subroutine for
C       MSIS-86 density model and GWS4 for thermospheric winds.
C        IYD - YEAR AND DAY AS YYDDD
C        SEC - UT(SEC)  (Not important in lower atmosphere)
C        ALT - ALTITUDE(KM) 
C        GLAT - GEODETIC LATITUDE(DEG)
C        GLONG - GEODETIC LONGITUDE(DEG)
C        STL - LOCAL APPARENT SOLAR TIME(HRS)
C        F107A - 3 MONTH AVERAGE OF F10.7 FLUX (Use 150 in lower atmos.)
C        F107 - DAILY F10.7 FLUX FOR PREVIOUS DAY ( " )
C        AP - Two element array with
C             AP(1) = MAGNETIC INDEX(DAILY) (use 4 in lower atmos.)
C             AP(2)=CURRENT 3HR ap INDEX (used only when SW(9)=-1.)
C     Note:  Ut, Local Time, and Longitude are used independently in the
C            model and are not of equal importance for every situation.  
C            For the most physically realistic calculation these three
C            variables should be consistent.
C      OUTPUT
C        W(1) = MERIDIONAL (m/sec + Northward)
C        W(2) = ZONAL (m/sec + Eastward)
C          ADDITIONAL COMMENTS
C               TO TURN ON AND OFF PARTICULAR VARIATIONS CALL TSELEC(SW)
C               WHERE SW IS A 25 ELEMENT ARRAY CONTAINING 0. FOR OFF, 1. 
C               FOR ON, OR 2. FOR MAIN EFFECTS OFF BUT CROSS TERMS ON
C               FOR THE FOLLOWING VARIATIONS
C               1 - F10.7 EFFECT ON MEAN  2 - TIME INDEPENDENT
C               3 - SYMMETRICAL ANNUAL    4 - SYMMETRICAL SEMIANNUAL
C               5 - ASYMMETRICAL ANNUAL   6 - ASYMMETRICAL SEMIANNUAL
C               7 - DIURNAL               8 - SEMIDIURNAL
C               9 - DAILY AP             10 - ALL UT/LONG EFFECTS
C              11 - LONGITUDINAL         12 - UT AND MIXED UT/LONG
C              13 - MIXED AP/UT/LONG     14 - TERDIURNAL
C              16 - ALL WINDF VAR        17 - ALL WZL VAR
C              18 - ALL UN1 VAR          19 - ALL WDZL VAR
C              24 - ALL B FIELDS (DIV)   25 - ALL C FIELDS (CURL)
C
C              To get current values of SW: CALL TRETRV(SW)
C
C             For example, to get zonal averages (no diurnal or
C             longitudinal variations) set SW(7),SW(8), SW(14),
C             and SW(10) equal to 0.  To just remove tidal variations 
C             set SW(7),SW(8), and SW(14) equal to 0.
      PARAMETER (MN1=5,MN2=14)
      DIMENSION AP(1),W(2),WINDF(2),WW(2),SV(25)
      DIMENSION WZL(2),WDZL(2)
      DIMENSION ZN1(MN1),UN1(MN1,2),UGN1(2,2)
      DIMENSION ZN2(MN2),UN2(MN2,2),UGN2(2,2)
      COMMON/PARMW5/PWB(200),PWC(200),PWBL(150),PWCL(150),PWBLD(150),
     $ PWCLD(150),PB12(150),PC12(150),PB13(150),PC13(150),
     $ PB14(150),PC14(150),PB15(150),PC15(150),
     $ PB15D(150),PC15D(150),PWP(100,26)
      COMMON/CSW/SW(25),ISW,SWC(25)               
      COMMON/HWMC/WBT(2),WCT(2)
      COMMON/DATW/ISD(3),IST(2),NAM(2)
      COMMON/DATIME/ISDATE(3),ISTIME(2),NAME(2)
      SAVE
      EXTERNAL INITW5,GWSBK5
      DATA S/.016/,ZL/200./,SV/25*1./,NNN/3/,MN2S/1/,MN2M/1/
      DATA ZN1/200.,150.,130.,115.,100./
      DATA ZN2/100.,90.,82.5,75.,67.5,60.,52.5,45.,37.5,30.,22.5,
     $ 15.,7.5,0/
C      Put identification data into common/datime/
      DO 1 I=1,3
        ISDATE(I)=ISD(I)
    1 CONTINUE
      DO 2 I=1,2
        ISTIME(I)=IST(I)
        NAME(I)=NAM(I)
    2 CONTINUE
      IF(ISW.NE.64999) CALL TSELEC(SV)
      YRD=IYD
      WW(1)=W(1)
      WW(2)=W(2)
C
      IF(ALT.LE.ZN1(MN1)) GOTO 50
C
C       EXOSPHERE WIND
      CALL GLBW5E(YRD,SEC,GLAT,GLONG,STL,F107A,F107,AP,PWB,PWC,WINDF)
      WINDF(1)=SW(16)*WINDF(1)
      WINDF(2)=SW(16)*WINDF(2)
C       WIND  AT ZL
      CALL GLBW5M(YRD,SEC,GLAT,GLONG,STL,F107A,F107,AP,PWBL,PWCL,WW)
      WZL(1)=(PWBL(1)*WINDF(1)+WW(1))*SW(17)*SW(18)
      WZL(2)=(PWBL(1)*WINDF(2)+WW(2))*SW(17)*SW(18)
      UN1(1,1)=WZL(1)
      UN1(1,2)=WZL(2)
C       WIND DERIVATIVE AT ZL
      WW(1)=0
      WW(2)=0
      CALL GLBW5M(YRD,SEC,GLAT,GLONG,STL,F107A,F107,AP,PWBLD,PWCLD,WW)
      WDZL(1)=(PWBLD(1)*WINDF(1)+WW(1))*SW(19)*SW(18)
      WDZL(2)=(PWBLD(1)*WINDF(2)+WW(2))*SW(19)*SW(18)
      UGN1(1,1)=WDZL(1)*S
      UGN1(1,2)=WDZL(2)*S
C
      IF(ALT.GE.ZL) GOTO 90
C
C        WIND AT ZN1(2) (150)
      CALL GLBW5M(YRD,SEC,GLAT,GLONG,STL,F107A,F107,AP,PB12,PC12,WW)
      UN1(2,1)=(PB12(1)*WINDF(1)+WW(1))*SW(18)
      UN1(2,2)=(PB12(1)*WINDF(2)+WW(2))*SW(18)
C        WIND AT ZN1(3) (130)
      CALL GLBW5M(YRD,SEC,GLAT,GLONG,STL,F107A,F107,AP,PB13,PC13,WW)
      UN1(3,1)=WW(1)*SW(18)
      UN1(3,2)=WW(2)*SW(18)
C        WIND AT ZN1(4) (115)
      CALL GLBW5M(YRD,SEC,GLAT,GLONG,STL,F107A,F107,AP,PB14,PC14,WW)
      UN1(4,1)=WW(1)*SW(18)
      UN1(4,2)=WW(2)*SW(18)
C
   50 CONTINUE
      MNN=MAX(1,MIN(MN2,NNN+1))
      IF(ALT.LT.ZN2(MNN)) GOTO 40
C
C        WIND AT ZN1(5) (100)
      CALL GLBW5M(YRD,SEC,GLAT,GLONG,STL,F107A,F107,AP,PB15,PC15,WW)
      UN1(5,1)=WW(1)*SW(18)
      UN1(5,2)=WW(2)*SW(18)
C         WIND DERIVATIVE AT ZN1(5) (100)
      CALL GLBW5M(YRD,SEC,GLAT,GLONG,STL,F107A,F107,AP,PB15D,PC15D,WW)
      UGN1(2,1)=WW(1)*SW(18)
      UGN1(2,2)=WW(2)*SW(18)
C
      IF(ALT.GE.ZN1(MN1)) GOTO 90
C
      UGN2(1,1)=UGN1(2,1)
      UGN2(1,2)=UGN1(2,2)
      UN2(1,1)=UN1(5,1)
      UN2(1,2)=UN1(5,2)
      GOTO 45
   40 CONTINUE
      UGN2(1,1)=1.E30
      UGN2(1,2)=1.E30
      UN2(1,1)=0
      UN2(1,2)=0
   45 CONTINUE
C
      DO 10 I=1,MN2
        IF(ALT.GT.ZN2(I)) GOTO 12
   10 CONTINUE
      I=MN2
   12 IZ=I
      MN2S=MAX(1,MIN(IZ-1,IZ-NNN))
      MN2E=MIN(MN2,MAX(MN2S+1,IZ-1+NNN))
      DO 20 I=MN2S,MN2E
        II=2*(I-2)+1
        IF(I.GT.1) THEN
          CALL GLBW5S(IYD,GLAT,GLONG,STL,PWP(1,II),PWP(1,II+1),WW)
          UN2(I,1)=WW(1)*SW(20)
          UN2(I,2)=WW(2)*SW(20)
        ENDIF
   20 CONTINUE
      MN2M=MN2E-MN2S+1
      UGN2(2,1)=1.E30
      UGN2(2,2)=1.E30
   90 CONTINUE
C       WIND AT ALTITUDE
      IF(W(1).NE.9898)
     $ W(1)= WPROF(ALT,ZL,S,WINDF(1),WZL(1),WDZL(1),
     $  MN1,ZN1,UN1(1,1),UGN1(1,1),MN2M,ZN2(MN2S),UN2(MN2S,1),UGN2(1,1))
      IF(W(2).NE.9898)
     $ W(2)= WPROF(ALT,ZL,S,WINDF(2),WZL(2),WDZL(2),
     $  MN1,ZN1,UN1(1,2),UGN1(1,2),MN2M,ZN2(MN2S),UN2(MN2S,2),UGN2(1,2))
      RETURN
C       Set number of nodes calculated each side of required altitude
C         to adjust profile accuracy vs efficiency
      ENTRY SETNW5(NNW)
      NNN=NNW
      END
C-----------------------------------------------------------------------
      FUNCTION WPROF(Z,ZL,S,UINF,ULB,ULBD,MN1,ZN1,UN1,UGN1,
     $   MN2,ZN2,UN2,UGN2)
      DIMENSION ZN1(MN1),UN1(MN1),UGN1(2),XS(15),YS(15),Y2OUT(15)
      DIMENSION ZN2(MN2),UN2(MN2),UGN2(2)
      SAVE
      IF(Z.GE.ZL) THEN
        X=S*(Z-ZL)
        F=EXP(-X)
        WPROF=UINF+(ULB-UINF)*F+(ULB-UINF+ULBD)*X*F
        RETURN
      ENDIF
      IF(Z.GE.ZN1(MN1).AND.Z.LT.ZN1(1)) THEN
        MN=MN1
        Z1=ZN1(1)
        Z2=ZN1(MN)
        ZDIF=Z2-Z1
        DO 10 K=1,MN
          XS(K)=(ZN1(K)-Z1)/ZDIF
          YS(K)=UN1(K)
   10   CONTINUE
        YD1=UGN1(1)*ZDIF
        YD2=UGN1(2)*ZDIF
        CALL SPLINE(XS,YS,MN,YD1,YD2,Y2OUT)
C      Eq.
        X=(Z-Z1)/ZDIF
C      Eq. 
        CALL SPLINT(XS,YS,Y2OUT,MN,X,Y)
        WPROF=Y
        RETURN
      ENDIF
      IF(Z.LT.ZN2(1)) THEN
        MN=MN2
        Z1=ZN2(1)
        Z2=ZN2(MN)
        ZDIF=Z2-Z1
        DO 20 K=1,MN
          XS(K)=(ZN2(K)-Z1)/ZDIF
          YS(K)=UN2(K)
   20   CONTINUE
        YD1=UGN2(1)
        IF(UGN2(1).LT.1.E30) YD1=UGN2(1)*ZDIF
        YD2=UGN2(2)
        IF(UGN2(2).LT.1.E30) YD2=UGN2(2)*ZDIF
        CALL SPLINE(XS,YS,MN,YD1,YD2,Y2OUT)
C      Eq.
        X=(Z-Z1)/ZDIF
C      Eq. 
        CALL SPLINT(XS,YS,Y2OUT,MN,X,Y)
        WPROF=Y
        RETURN
      ENDIF
      RETURN
      END
C-----------------------------------------------------------------------
      SUBROUTINE GLBW5E(YRD,SEC,LAT,LONG,STL,F107A,F107,AP,PB,PC,WW)
      REAL LAT,LONG
      DIMENSION WB(2,15),WC(2,15),PB(1),PC(1),WW(2)
      DIMENSION AP(1)
      COMMON/CSW/SW(25),ISW,SWC(25)
      COMMON/HWMC/WBT(2),WCT(2)
C      COMMON/VPOLY/BT(20,20),BP(20,20),CSTL,SSTL,C2STL,S2STL,
C     $ C3STL,S3STL,IYR,DAY,DF,DFA,DFC,APD,APDF,APDFC,APT,SLT
      COMMON/VPOLY2/XVL,LVL,MVL,CLAT,SLAT,BT(20,20),BP(20,20)
      COMMON/LTCOMP/TLL,NSVL,CSTL,SSTL,C2STL,S2STL,C3STL,S3STL
      COMMON/LGCOMP/XLL,NGVL,CLONG,SLONG,C2LONG,S2LONG
      SAVE
      DATA DGTR/.017453/,SR/7.2722E-5/,HR/.2618/,DR/1.72142E-2/
      DATA NSW/14/,WB/30*0/,WC/30*0/
      DATA PB14/-1./,PB18/-1./
      DATA SW9/1./,LV/12/,MV/3/,NSV/3/,NGV/2/,PSET/3./
      G0(A)=(A-4.+(PB(26)-1.)*(A-4.+(EXP(-ABS(PB(25))*(A-4.))-1.)/
     * ABS(PB(25))))
C       CONFIRM PARAMETER SET
      IF(PB(100).EQ.0) PB(100)=PSET
      IF(PB(100).NE.PSET) THEN
        WRITE(6,900) PB(100),PC(100)
  900   FORMAT(1X,'WRONG PARAMETER SET FOR GLBW5E',3F10.1)
        STOP
      ENDIF
C
      DO 10 J=1,NSW
        WB(1,J)=0
        WB(2,J)=0
        WC(1,J)=0
        WC(2,J)=0
   10 CONTINUE
      IF(SW(9).GT.0) SW9=1.
      IF(SW(9).LT.0) SW9=-1.
      IYR = YRD/1000.
      DAY = YRD - IYR*1000.
      IF(XVL.NE.LAT.OR.LV.GT.LVL.OR.MV.GT.MVL) THEN
        SLAT=SIN(DGTR*LAT)
        CLAT=COS(DGTR*LAT)
        CALL VSPHR1(SLAT,CLAT,LV,MV,BT,BP,20)
        XVL=LAT
        LVL=LV
        MVL=MV
      ENDIF
      IF(TLL.NE.STL.OR.NSV.GT.NSVL)  THEN
        SSTL = SIN(HR*STL)
        CSTL = COS(HR*STL)
        S2STL = SIN(2.*HR*STL)
        C2STL = COS(2.*HR*STL)
        S3STL = SIN(3.*HR*STL)
        C3STL = COS(3.*HR*STL)
        TLL = STL
        NSVL=NSV
      ENDIF
      IF(DAY.NE.DAYL.OR.PB(14).NE.PB14) THEN
        CD14=COS(DR*(DAY-PB(14)))
C        SD14=SIN(DR*(DAY-PB(14)))
      ENDIF
      IF(DAY.NE.DAYL.OR.PB(18).NE.PB18) CD18=COS(2.*DR*(DAY-PB(18)))
      DAYL=DAY
      PB14=PB(14)
      PB18=PB(18)
      IF(XLL.NE.LONG) THEN
        SLONG=SIN(DGTR*LONG)
        CLONG=COS(DGTR*LONG)
        S2LONG=SIN(2.*DGTR*LONG)
        C2LONG=COS(2.*DGTR*LONG)
        XLL=LONG
        NGVL=2
      ENDIF
C       F10.7 EFFECT
      DF=F107-F107A
      DFA=F107A-150.
      DFC=DFA+PB(20)*DF
C       TIME INDEPENDENT
      F1B=1.+PB(22)*DFC*SWC(1)
      IF(WW(1).NE.9898) THEN
       WB(1,2)=(PB(2)*BT(3,1)+PB(3)*BT(5,1)+PB(23)*BT(7,1))*F1B
      ENDIF
      WB(2,2)=0.
      F1C=1.+PC(22)*DFC*SWC(1)
      WC(1,2)=0.
      IF(WW(2).NE.9898) THEN
       WC(2,2)=-(PC(2)*BT(2,1)+PC(3)*BT(4,1)+PC(23)*BT(6,1))*F1C
     $ -(PC(27)*BT(3,1)+PC(15)*BT(5,1)+PC(60)*BT(7,1)
     $ +PC(161)*BT(9,1)+PC(162)*BT(11,1)+PC(163)*BT(13,1))*F1C
      ENDIF
C       SYMMETRICAL ANNUAL
C       SYMMETRICAL SEMIANNUAL
      IF(WW(1).NE.9898) THEN
       WB(1,4)=(PB(17)*BT(3,1)+PB(31)*BT(5,1))*CD18
      ENDIF
      WB(2,4)=0
      WC(1,4)=0
      IF(WW(2).NE.9898) THEN
       WC(2,4)=-(PC(17)*BT(2,1)+PC(31)*BT(4,1))*CD18
      ENDIF
C       ASYMMETRICAL ANNUAL
      F5B=1.+PB(48)*DFC*SWC(1)
      IF(WW(1).NE.9898) THEN
       WB(1,5)=(PB(10)*BT(2,1)+PB(11)*BT(4,1))*CD14*F5B
      ENDIF
      WB(2,5)=0
      F5C=1.+PC(48)*DFC*SWC(1)
      WC(1,5)=0
      IF(WW(2).NE.9898) THEN
       WC(2,5)=-(PC(10)*BT(3,1)+PC(11)*BT(5,1))*CD14*F5C
      ENDIF
C       ASYMMETRICAL SEMIANNUAL
C         none
C       DIURNAL      
      IF(SW(7).EQ.0) GOTO 200
      F7B=1.+PB(50)*DFC*SWC(1)
      F75B=1.+PB(83)*DFC*SWC(1)
      IF(WW(1).NE.9898) THEN
       WB(1,7)=(PB(7)*BT(2,2)+PB(8)*BT(4,2)+PB(29)*BT(6,2)
     $ +PB(142)*BT(8,2)+PB(144)*BT(10,2)
     $  +PB(182)*BT(3,2)+PB(184)*BT(5,2)
     $  )*SSTL*F7B
     $ +(PB(13)*BT(3,2)+PB(146)*BT(5,2))
     $    *CD14*SSTL*F75B*SWC(5)
     $ +(PB(171)*BT(2,2)+PB(173)*BT(4,2))
     $    *CD18*SSTL*F75B*SWC(4)
     $ + (PB(4)*BT(2,2)+PB(5)*BT(4,2)+PB(28)*BT(6,2)
     $ +PB(141)*BT(8,2)+PB(143)*BT(10,2)
     $  +PB(181)*BT(3,2)+PB(183)*BT(5,2)
     $  )*CSTL*F7B
     $ +(PB(12)*BT(3,2)+PB(145)*BT(5,2))
     $      *CD14*CSTL*F75B*SWC(5)
     $ +(PB(170)*BT(2,2)+PB(172)*BT(4,2))
     $    *CD18*CSTL*F75B*SWC(4)
      ENDIF
      IF(WW(2).NE.9898) THEN
       WB(2,7)=-(PB(4)*BP(2,2)+PB(5)*BP(4,2)+PB(28)*BP(6,2)
     $   +PB(141)*BP(8,2)+PB(143)*BP(10,2)
     $   +PB(181)*BP(3,2)+PB(183)*BP(5,2)
     $  )*SSTL*F7B
     $ -(PB(12)*BP(3,2)+PB(145)*BP(5,2))
     $    *CD14*SSTL*F75B*SWC(5)
     $ -(PB(170)*BP(2,2)+PB(172)*BP(4,2))
     $    *CD18*SSTL*F75B*SWC(4)
     $ + (PB(7)*BP(2,2)+PB(8)*BP(4,2)+PB(29)*BP(6,2)
     $   +PB(142)*BP(8,2)+PB(144)*BP(10,2)
     $   +PB(182)*BP(3,2)+PB(184)*BP(5,2)
     $  )*CSTL*F7B
     $ +(PB(13)*BP(3,2)+PB(146)*BP(5,2))
     $    *CD14*CSTL*F75B*SWC(5)
     $ +(PB(171)*BP(2,2)+PB(173)*BP(4,2))
     $    *CD18*CSTL*F75B*SWC(4)
      ENDIF
      F7C=1.+PC(50)*DFC*SWC(1)
      F75C=1.+PC(83)*DFC*SWC(1)
      IF(WW(1).NE.9898) THEN
       WC(1,7)=-(PC(4)*BP(3,2)+PC(5)*BP(5,2)+PC(28)*BP(7,2)
     $   +PC(141)*BP(9,2)+PC(143)*BP(11,2)
     $   +PC(181)*BP(2,2)+PC(183)*BP(4,2)+PC(185)*BP(6,2)
     $   +PC(187)*BP(8,2)+PC(189)*BP(10,2)
     $  )*SSTL*F7C
     $ -(PC(12)*BP(2,2)+PC(145)*BP(4,2))
     $    *CD14*SSTL*F75C*SWC(5)
     $ -(PC(170)*BP(3,2)+PC(172)*BP(5,2))
     $    *CD18*SSTL*F75C*SWC(4)
     $ +(PC(7)*BP(3,2)+PC(8)*BP(5,2)+PC(29)*BP(7,2)
     $ +PC(142)*BP(9,2)+PC(144)*BP(11,2)
     $ +PC(182)*BP(2,2)+PC(184)*BP(4,2)+PC(186)*BP(6,2)
     $ +PC(188)*BP(8,2)+PC(190)*BP(10,2)
     $  )*CSTL*F7C
     $ +(PC(13)*BP(2,2)+PC(146)*BP(4,2))
     $     *CD14*CSTL*F75C*SWC(5)
     $ +(PC(171)*BP(3,2)+PC(173)*BP(5,2))
     $    *CD18*CSTL*F75C*SWC(4)
      ENDIF
      IF(WW(2).NE.9898) THEN
       WC(2,7)=-(PC(7)*BT(3,2)+PC(8)*BT(5,2)+PC(29)*BT(7,2)
     $ +PC(142)*BT(9,2)+PC(144)*BT(11,2)
     $ +PC(182)*BT(2,2)+PC(184)*BT(4,2)+PC(186)*BT(6,2)
     $ +PC(188)*BT(8,2)+PC(190)*BT(10,2)
     $  )*SSTL*F7C
     $ -(PC(13)*BT(2,2)+PC(146)*BT(4,2))
     $    *CD14*SSTL*F75C*SWC(5)
     $ -(PC(171)*BT(3,2)+PC(173)*BT(5,2))
     $    *CD18*SSTL*F75C*SWC(4)
     $ -(PC(4)*BT(3,2)+PC(5)*BT(5,2)+PC(28)*BT(7,2)
     $ +PC(141)*BT(9,2)+PC(143)*BT(11,2)
     $ +PC(181)*BT(2,2)+PC(183)*BT(4,2)+PC(185)*BT(6,2)
     $ +PC(187)*BT(8,2)+PC(189)*BT(10,2)
     $  )*CSTL*F7C
     $ -(PC(12)*BT(2,2)+PC(145)*BT(4,2))
     $    *CD14*CSTL*F75C*SWC(5)
     $ -(PC(170)*BT(3,2)+PC(172)*BT(5,2))
     $    *CD18*CSTL*F75C*SWC(4)
      ENDIF
  200 CONTINUE
C       SEMIDIURNAL
      IF(SW(8).EQ.0) GOTO 210
      F8B=1.+PB(90)*DFC*SWC(1)
      IF(WW(1).NE.9898) THEN
       WB(1,8)=(PB(9)*BT(3,3)+PB(43)*BT(5,3)
     $   +PB(111)*BT(7,3)
     $   +(PB(34)*BT(4,3)+PB(148)*BT(6,3))*CD14*SWC(5)
     $   +(PB(134)*BT(3,3))*CD18*SWC(4) 
     $   +PB(152)*BT(4,3)+PB(154)*BT(6,3)+PB(156)*BT(8,3)
     $   +PB(158)*BT(10,3)
     $  )*S2STL*F8B
     $ +(PB(6)*BT(3,3)+PB(42)*BT(5,3)
     $   +PB(110)*BT(7,3)
     $   +(PB(24)*BT(4,3)+PB(147)*BT(6,3))*CD14*SWC(5)
     $   +(PB(135)*BT(3,3))*CD18*SWC(4)
     $   +PB(151)*BT(4,3)+PB(153)*BT(6,3)+PB(155)*BT(8,3)
     $   +PB(157)*BT(10,3)
     $  )*C2STL*F8B
      ENDIF
      IF(WW(2).NE.9898) THEN
       WB(2,8)=-(PB(6)*BP(3,3)+PB(42)*BP(5,3)
     $   +PB(110)*BP(7,3)
     $   +(PB(24)*BP(4,3)+PB(147)*BP(6,3))*CD14*SWC(5)
     $   +(PB(135)*BP(3,3))*CD18*SWC(4)
     $   +PB(151)*BP(4,3)+PB(153)*BP(6,3)+PB(155)*BP(8,3)
     $   +PB(157)*BP(10,3)
     $  )*S2STL*F8B
     $   + (PB(9)*BP(3,3)+PB(43)*BP(5,3)
     $   +PB(111)*BP(7,3)
     $   +(PB(34)*BP(4,3)+PB(148)*BP(6,3))*CD14*SWC(5)
     $   +(PB(134)*BP(3,3))*CD18*SWC(4)
     $   +PB(152)*BP(4,3)+PB(154)*BP(6,3)+PB(156)*BP(8,3)
     $   +PB(158)*BP(10,3)
     $  )*C2STL*F8B
      ENDIF
      F8C=1.+PC(90)*DFC*SWC(1)
      IF(WW(1).NE.9898) THEN
       WC(1,8)=-(PC(6)*BP(4,3)+PC(42)*BP(6,3)
     $   +PC(110)*BP(8,3)
     $   +(PC(24)*BP(3,3)+PC(147)*BP(5,3))*CD14*SWC(5)
     $   +(PC(135)*BP(4,3))*CD18*SWC(4)
     $   +PC(151)*BP(3,3)+PC(153)*BP(5,3)+PC(155)*BP(7,3)
     $   +PC(157)*BP(9,3)
     $  )*S2STL*F8C
     $ +(PC(9)*BP(4,3)+PC(43)*BP(6,3)
     $   +PC(111)*BP(8,3)
     $   +(PC(34)*BP(3,3)+PC(148)*BP(5,3))*CD14*SWC(5)
     $   +(PC(134)*BP(4,3))*CD18*SWC(4)
     $   +PC(152)*BP(3,3)+PC(154)*BP(5,3)+PC(156)*BP(7,3)
     $   +PC(158)*BP(9,3)
     $  )*C2STL*F8C
      ENDIF
      IF(WW(2).NE.9898) THEN
       WC(2,8)=-(PC(9)*BT(4,3)+PC(43)*BT(6,3)
     $   +PC(111)*BT(8,3)
     $   +(PC(34)*BT(3,3)+PC(148)*BT(5,3))*CD14*SWC(5)
     $   +(PC(134)*BT(4,3))*CD18*SWC(4)
     $   +PC(152)*BT(3,3)+PC(154)*BT(5,3)+PC(156)*BT(7,3)
     $   +PC(158)*BT(9,3)
     $  )*S2STL*F8C
     $ - (PC(6)*BT(4,3)+PC(42)*BT(6,3)
     $   +PC(110)*BT(8,3)
     $   +(PC(24)*BT(3,3)+PC(147)*BT(5,3))*CD14*SWC(5)
     $   +(PC(135)*BT(4,3))*CD18*SWC(4)
     $   +PC(151)*BT(3,3)+PC(153)*BT(5,3)+PC(155)*BT(7,3)
     $   +PC(157)*BT(9,3)
     $  )*C2STL*F8C
      ENDIF
  210 CONTINUE
C        TERDIURNAL
      IF(SW(14).EQ.0) GOTO 220
      F14B=1.
      IF(WW(1).NE.9898) THEN
       WB(1,14)=(PB(40)*BT(4,4)+PB(149)*BT(6,4)
     $   +PB(114)*BT(8,4)
     $   +(PB(94)*BT(5,4)+PB(47)*BT(7,4))*CD14*SWC(5)
     $  )*S3STL*F14B
     $ + (PB(41)*BT(4,4)+PB(150)*BT(6,4)
     $   +PB(115)*BT(8,4)
     $   +(PB(95)*BT(5,4)+PB(49)*BT(7,4))*CD14*SWC(5)
     $  )*C3STL*F14B
      ENDIF
      IF(WW(2).NE.9898) THEN
       WB(2,14)=-(PB(41)*BP(4,4)+PB(150)*BP(6,4)
     $   +PB(115)*BP(8,4)
     $   +(PB(95)*BP(5,4)+PB(49)*BP(7,4))*CD14*SWC(5)
     $  )*S3STL*F14B
     $ + (PB(40)*BP(4,4)+PB(149)*BP(6,4)
     $   +PB(114)*BP(8,4)
     $   +(PB(94)*BP(5,4)+PB(47)*BP(7,4))*CD14*SWC(5)
     $  )*C3STL*F14B
      ENDIF
      F14C=1.
      IF(WW(1).NE.9898) THEN
       WC(1,14)=-(PC(41)*BP(5,4)+PC(150)*BP(7,4)
     $   +PC(115)*BP(9,4)
     $   +(PC(95)*BP(4,4)+PC(49)*BP(6,4))*CD14*SWC(5)
     $  )*S3STL*F14C
     $ + (PC(40)*BP(5,4)+PC(149)*BP(7,4)
     $   +PC(114)*BP(9,4)
     $   +(PC(94)*BP(4,4)+PC(47)*BP(6,4))*CD14*SWC(5)
     $  )*C3STL*F14C
      ENDIF
      IF(WW(2).NE.9898) THEN
       WC(2,14)=-(PC(40)*BT(5,4)+PC(149)*BT(7,4)
     $   +PC(114)*BT(9,4)
     $   +(PC(94)*BT(4,4)+PC(47)*BT(6,4))*CD14*SWC(5)
     $  )*S3STL*F14C
     $ - (PC(41)*BT(5,4)+PC(150)*BT(7,4)
     $   +PC(115)*BT(9,4)
     $   +(PC(95)*BT(4,4)+PC(49)*BT(6,4))*CD14*SWC(5)
     $  )*C3STL*F14C
      ENDIF
  220 CONTINUE
C        MAGNETIC ACTIVITY
      IF(SW(9).EQ.0.) GOTO 40
      IF(SW9.EQ.-1.) GOTO 30
C           daily AP
      APD=AP(1)-4.
      APDF=(APD+(PB(45)-1.)*(APD+(EXP(-PB(44)*APD)-1.)/PB(44)))
C      APDFC=(APD+(PC(45)-1.)*(APD+(EXP(-PC(44)*APD)-1.)/PC(44)))
      APDFC=APDF
      IF(APD.EQ.0.) GOTO 40
      IF(WW(1).NE.9898) THEN
       WB(1,9)=(PB(46)*BT(3,1)+PB(35)*BT(5,1)+PB(33)*BT(7,1))*APDF
     $  +(PB(175)*BT(3,3)+PB(177)*BT(5,3))*S2STL*APDF
     $  +(PB(174)*BT(3,3)+PB(176)*BT(5,3))*C2STL*APDF
      ENDIF
      IF(WW(2).NE.9898) THEN
       WB(2,9)=0                                              
     $  -(PB(174)*BP(3,3)+PB(176)*BP(5,3))*S2STL*APDF
     $  +(PB(175)*BP(3,3)+PB(177)*BP(5,3))*C2STL*APDF
      ENDIF
      IF(WW(1).NE.9898) THEN
       WC(1,9)=SWC(7)*WC(1,7)*PC(122)*APDFC
     $  -(PC(174)*BP(4,3)+PC(176)*BP(6,3))*S2STL*APDFC
     $  +(PC(175)*BP(4,3)+PC(177)*BP(6,3))*C2STL*APDFC
      ENDIF
      IF(WW(2).NE.9898) THEN
       WC(2,9)=-(PC(46)*BT(2,1)+PC(35)*BT(4,1)+PC(33)*BT(6,1))*APDFC
     $ +SWC(7)*WC(2,7)*PC(122)*APDFC
     $ -(PC(175)*BT(4,3)+PC(177)*BT(6,3))*S2STL*APDFC
     $ -(PC(174)*BT(4,3)+PC(176)*BT(6,3))*C2STL*APDFC
      ENDIF
      GO TO 40
   30 CONTINUE
      IF(PB(25).LT.1.E-4) PB(25)=1.E-4
      APT=G0(AP(2))
      IF(APT.EQ.0) GOTO 40
      IF(WW(1).NE.9898) THEN
       WB(1,9)=(PB(97)*BT(3,1)+PB(55)*BT(5,1)+PB(51)*BT(7,1))*APT
     $  +(PB(160)*BT(3,3)+PB(179)*BT(5,3))*S2STL*APT
     $  +(PB(159)*BT(3,3)+PB(178)*BT(5,3))*C2STL*APT
      ENDIF
      IF(WW(2).NE.9898) THEN
       WB(2,9)=0
     $  -(PB(159)*BP(3,3)+PB(178)*BP(5,3))*S2STL*APT
     $  +(PB(160)*BP(3,3)+PB(179)*BP(5,3))*C2STL*APT
      ENDIF
      IF(WW(1).NE.9898) THEN
       WC(1,9)=SWC(7)*WC(1,7)*PC(129)*APT
     $  -(PC(159)*BP(4,3)+PC(178)*BP(6,3))*S2STL*APT
     $  +(PC(160)*BP(4,3)+PC(179)*BP(6,3))*C2STL*APT
      ENDIF
      IF(WW(2).NE.9898) THEN
      WC(2,9)=-(PC(97)*BT(2,1)+PC(55)*BT(4,1)+PC(51)*BT(6,1))*APT
     $ +SWC(7)*WC(2,7)*PC(129)*APT
     $ -(PC(160)*BT(4,3)+PC(179)*BT(6,3))*S2STL*APT
     $ -(PC(159)*BT(4,3)+PC(178)*BT(6,3))*C2STL*APT
      ENDIF
  40  CONTINUE
      IF(SW(10).EQ.0) GOTO 49
C        LONGITUDINAL
      DBASY1=1.+PB(199)*SLAT
      DBASY2=1.+PB(200)*SLAT
      F11B=1.+PB(81)*DFC*SWC(1)
      IF(SW(11).EQ.0) GOTO 230
      IF(WW(1).NE.9898) THEN
       WB(1,11)=(PB(91)*BT(3,2)+PB(92)*BT(5,2)+PB(93)*BT(7,2))
     $  *SLONG*DBASY1*F11B
     $ + (PB(65)*BT(3,2)+PB(66)*BT(5,2)+PB(67)*BT(7,2))
     $  *CLONG*DBASY1*F11B
     $  +(PB(191)*BT(3,3)+PB(193)*BT(5,3)+PB(195)*BT(7,3)
     $   +PB(197)*BT(9,3)
     $  )*S2LONG*DBASY2*F11B
     $ + (PB(192)*BT(3,3)+PB(194)*BT(5,3)+PB(196)*BT(7,3)
     $    +PB(198)*BT(9,3)
     $  )*C2LONG*DBASY2*F11B
      ENDIF
      IF(WW(2).NE.9898) THEN
       WB(2,11)=-(PB(65)*BP(3,2)+PB(66)*BP(5,2)+PB(67)*BP(7,2))
     $  *SLONG*DBASY1*F11B
     $ + (PB(91)*BP(3,2)+PB(92)*BP(5,2)+PB(93)*BP(7,2))
     $  *CLONG*DBASY1*F11B
     $ -(PB(192)*BP(3,3)+PB(194)*BP(5,3)+PB(196)*BP(7,3)
     $   +PB(198)*BP(9,3)
     $  )*S2LONG*DBASY2*F11B
     $ + (PB(191)*BP(3,3)+PB(193)*BP(5,3)+PB(195)*BP(7,3)
     $    +PB(197)*BP(9,3)
     $  )*C2LONG*DBASY2*F11B
      ENDIF
      DCASY1=1.+PC(199)*SLAT
      DCASY2=1.+PC(200)*SLAT
      F11C=1.+PC(81)*DFC*SWC(1)
      IF(WW(1).NE.9898) THEN
       WC(1,11)=-(PC(65)*BP(2,2)+PC(66)*BP(4,2)+PC(67)*BP(6,2)
     $ +PC(73)*BP(8,2)+PC(74)*BP(10,2)
     $  )*SLONG*DCASY1*F11C
     $ + (PC(91)*BP(2,2)+PC(92)*BP(4,2)+PC(93)*BP(6,2)
     $ +PC(87)*BP(8,2)+PC(88)*BP(10,2)
     $  )*CLONG*DCASY1*F11C
     $  -(PC(192)*BP(4,3)+PC(194)*BP(6,3)+PC(196)*BP(8,3)
     $ +PC(198)*BP(10,3)
     $  )*S2LONG*DCASY2*F11C
     $ + (PC(191)*BP(4,3)+PC(193)*BP(6,3)+PC(195)*BP(8,3)
     $ +PC(197)*BP(10,3)
     $  )*C2LONG*DCASY2*F11C
      ENDIF
      IF(WW(2).NE.9898) THEN
       WC(2,11)=-(PC(91)*BT(2,2)+PC(92)*BT(4,2)+PC(93)*BT(6,2)
     $ +PC(87)*BT(8,2)+PC(88)*BT(10,2)
     $  )*SLONG*DCASY1*F11C
     $ - (PC(65)*BT(2,2)+PC(66)*BT(4,2)+PC(67)*BT(6,2)
     $ +PC(73)*BT(8,2)+PC(74)*BT(10,2)
     $  )*CLONG*DCASY1*F11C
     $  -(PC(191)*BT(4,3)+PC(193)*BT(6,3)+PC(195)*BT(8,3)
     $ +PC(197)*BT(10,3)
     $  )*S2LONG*DCASY2*F11C
     $ - (PC(192)*BT(4,3)+PC(194)*BT(6,3)+PC(196)*BT(8,3)
     $ +PC(198)*BT(10,3)
     $  )*C2LONG*DCASY2*F11C
      ENDIF
  230 CONTINUE
C       UT & MIXED UT/LONG
      UTBASY=1.
      F12B=1.+PB(82)*DFC*SWC(1)
      IF(SW(12).EQ.0) GOTO 240
      IF(WW(1).NE.9898) THEN
       WB(1,12)=(PB(69)*BT(2,1)+PB(70)*BT(4,1)+PB(71)*BT(6,1)
     $ +PB(116)*BT(8,1)+PB(117)*BT(10,1)+PB(118)*BT(12,1)
     $  )*COS(SR*(SEC-PB(72)))*UTBASY*F12B
     $ + (PB(77)*BT(4,3)+PB(78)*BT(6,3)+PB(79)*BT(8,3))
     $  *COS(SR*(SEC-PB(80))+2.*DGTR*LONG)*UTBASY*F12B*SWC(11)
      ENDIF
      IF(WW(2).NE.9898) THEN
       WB(2,12)=(PB(77)*BP(4,3)+PB(78)*BP(6,3)+PB(79)*BP(8,3))
     $  *COS(SR*(SEC-PB(80)+21600.)+2.*DGTR*LONG)
     $    *UTBASY*F12B*SWC(11)
      ENDIF
      UTCASY=1.
      F12C=1.+PC(82)*DFC*SWC(1)
      IF(WW(1).NE.9898) THEN
       WC(1,12)=(PC(77)*BP(3,3)+PC(78)*BP(5,3)+PC(79)*BP(7,3)
     $ +PC(165)*BP(9,3)+PC(166)*BP(11,3)+PC(167)*BP(13,3)
     $  )*COS(SR*(SEC-PC(80))+2.*DGTR*LONG)*UTCASY*F12C*SWC(11)
      ENDIF
      IF(WW(2).NE.9898) THEN
       WC(2,12)=-(PC(69)*BT(3,1)+PC(70)*BT(5,1)+PC(71)*BT(7,1)
     $ +PC(116)*BT(9,1)+PC(117)*BT(11,1)+PC(118)*BT(13,1)
     $  )*COS(SR*(SEC-PC(72)))*UTCASY*F12C
     $ + (PC(77)*BT(3,3)+PC(78)*BT(5,3)+PC(79)*BT(7,3)
     $ +PC(165)*BT(9,3)+PC(166)*BT(11,3)+PC(167)*BT(13,3)
     $  )*COS(SR*(SEC-PC(80)+21600.)+2.*DGTR*LONG)
     $   *UTCASY*F12C*SWC(11)
      ENDIF
  240 CONTINUE
C       MIXED LONG,UT,AP
      IF(SW(13).EQ.0) GOTO 48
      IF(SW9.EQ.-1.) GO TO 45
      IF(APD.EQ.0) GOTO 48
      IF(WW(1).NE.9898) THEN
       WB(1,13)=
     $ (PB(61)*BT(3,2)+PB(62)*BT(5,2)+PB(63)*BT(7,2))
     $  *COS(DGTR*(LONG-PB(64)))*APDF*SWC(11)+
     $  (PB(84)*BT(2,1)+PB(85)*BT(4,1)+PB(86)*BT(6,1))
     $  *COS(SR*(SEC-PB(76)))*APDF*SWC(12)
      ENDIF
      IF(WW(2).NE.9898) THEN
       WB(2,13)=(PB(61)*BP(3,2)+PB(62)*BP(5,2)+PB(63)*BP(7,2))
     $  *COS(DGTR*(LONG-PB(64)+90.))*APDF*SWC(11)
      ENDIF
      IF(WW(1).NE.9898) THEN 
       WC(1,13)=SWC(11)*WC(1,11)*PC(61)*APDFC
     $ +SWC(12)*WC(1,12)*PC(84)*APDFC
      ENDIF
      IF(WW(2).NE.9898) THEN
       WC(2,13)=SWC(11)*WC(2,11)*PC(61)*APDFC
     $ +SWC(12)*WC(2,12)*PC(84)*APDFC
      ENDIF
      GOTO 48
   45 CONTINUE
      IF(APT.EQ.0) GOTO 48
      IF(WW(1).NE.9898) THEN
       WB(1,13)=
     $  (PB(53)*BT(3,2)+PB(99)*BT(5,2)+PB(68)*BT(7,2))
     $  *COS(DGTR*(LONG-PB(98)))*APT*SWC(11)+
     $  (PB(56)*BT(2,1)+PB(57)*BT(4,1)+PB(58)*BT(6,1))
     $  *COS(SR*(SEC-PB(59)))*APT*SWC(12)
      ENDIF
      IF(WW(2).NE.9898) THEN
       WB(2,13)=(PB(53)*BP(3,2)+PB(99)*BP(5,2)+PB(68)*BP(7,2))
     $  *COS(DGTR*(LONG-PB(98)+90.))*APT*SWC(11)
      ENDIF
      IF(WW(1).NE.9898) THEN
       WC(1,13)=SWC(11)*WC(1,11)*PC(53)*APT
     $ +SWC(12)*WC(1,12)*PC(56)*APT
      ENDIF
      IF(WW(2).NE.9898) THEN
       WC(2,13)=SWC(11)*WC(2,11)*PC(53)*APT
     $ +SWC(12)*WC(2,12)*PC(56)*APT
      ENDIF
   48 CONTINUE
   49 CONTINUE
      WBT(1)=0             
      WBT(2)=0
      WCT(1)=0
      WCT(2)=0                                 
C       SUM WINDS AND CHANGE MERIDIONAL SIGN TO + NORTH
      DO 50 K=1,NSW
        WBT(1)=WBT(1)-ABS(SW(K))*WB(1,K)
        WCT(1)=WCT(1)-ABS(SW(K))*WC(1,K)
        WBT(2)=WBT(2)+ABS(SW(K))*WB(2,K)
        WCT(2)=WCT(2)+ABS(SW(K))*WC(2,K)
   50 CONTINUE
      IF(WW(1).NE.9898) WW(1)=WBT(1)*SW(24)+WCT(1)*SW(25)
      IF(WW(2).NE.9898) WW(2)=WBT(2)*SW(24)+WCT(2)*SW(25)
      RETURN
      END
C-----------------------------------------------------------------------
      SUBROUTINE GLBW5M(YRD,SEC,LAT,LONG,STL,F107A,F107,AP,PB,PC,WW)
      REAL LAT,LONG
      DIMENSION WB(2,15),WC(2,15),PB(1),PC(1),WW(2)
      DIMENSION AP(1)
      COMMON/CSW/SW(25),ISW,SWC(25)
      COMMON/HWMC/WBT(2),WCT(2)
C      COMMON/VPOLY/BT(20,20),BP(20,20),CSTL,SSTL,C2STL,S2STL,
C     $ C3STL,S3STL,IYR,DAY,DF,DFA,DFC,APD,APDF,APDFC,APT,STL
      COMMON/VPOLY2/XVL,LVL,MVL,CLAT,SLAT,BT(20,20),BP(20,20)
      COMMON/LTCOMP/TLL,NSVL,CSTL,SSTL,C2STL,S2STL,C3STL,S3STL
      COMMON/LGCOMP/XLL,NGVL,CLONG,SLONG,C2LONG,S2LONG
      SAVE
      DATA DGTR/.017453/,SR/7.2722E-5/,HR/.2618/,DR/1.72142E-2/
      DATA PB14/-1./,PB18/-1./
      DATA NSW/14/,WB/30*0/,WC/30*0/
      DATA SW9/1./,LV/10/,MV/2/,NSV/2/,PSET/4./
      G0(A)=(A-4.+(PB(26)-1.)*(A-4.+(EXP(-ABS(PB(25))*(A-4.))-1.)/
     * ABS(PB(25))))
C       CONFIRM PARAMETER SET
      IF(PB(100).EQ.0) PB(100)=PSET
      IF(PB(100).NE.PSET) THEN
        WRITE(6,900) PSET,PB(100),PC(100)
  900   FORMAT(1X,'WRONG PARAMETER SET FOR GLBW5M',3F10.1)
        STOP
      ENDIF
C
      DO 10 J=1,NSW
        WB(1,J)=0
        WB(2,J)=0
        WC(1,J)=0
        WC(2,J)=0
   10 CONTINUE
      IF(SW(9).GT.0) SW9=1.
      IF(SW(9).LT.0) SW9=-1.
      IYR = YRD/1000.
      DAY = YRD - IYR*1000.
      IF(XVL.NE.LAT.OR.LV.GT.LVL.OR.MV.GT.MVL) THEN
        SLAT=SIN(DGTR*LAT)
        CLAT=COS(DGTR*LAT)
        CALL VSPHR1(SLAT,CLAT,LV,MV,BT,BP,20)
        XVL=LAT
        LVL=LV
        MVL=MV
      ENDIF
      IF(TLL.NE.STL.OR.NSV.GT.NSVL)  THEN
        SSTL = SIN(HR*STL)
        CSTL = COS(HR*STL)
        S2STL = SIN(2.*HR*STL)
        C2STL = COS(2.*HR*STL)
        TLL = STL
        NSVL=NSV
      ENDIF
      IF(DAY.NE.DAYL.OR.PB(14).NE.PB14) CD14=COS(DR*(DAY-PB(14)))
      IF(DAY.NE.DAYL.OR.PB(18).NE.PB18) CD18=COS(2.*DR*(DAY-PB(18)))
      IF(DAY.NE.DAYL.OR.PB(19).NE.PB19) CD19B=COS(2.*DR*(DAY-PB(19)))
      DAYL=DAY
      PB14=PB(14)
      PB18=PB(18)
      PB19=PB(19)
C       F10.7 EFFECT
      DF=F107-F107A
      DFA=F107A-150.
      DFC=DFA+PB(20)*DF
C       TIME INDEPENDENT
      F1B=1.
      IF(WW(1).NE.9898) THEN
       WB(1,2)=(PB(2)*BT(3,1)+PB(3)*BT(5,1)+PB(23)*BT(7,1))*F1B
      ENDIF
      WB(2,2)=0.
      F1C=1.
      WC(1,2)=0.
      IF(WW(2).NE.9898) THEN
       WC(2,2)=-(PC(2)*BT(2,1)+PC(3)*BT(4,1)+PC(23)*BT(6,1))*F1C
     $ -(PC(27)*BT(3,1)+PC(15)*BT(5,1)+PC(60)*BT(7,1))*F1C
      ENDIF
C       SYMMETRICAL ANNUAL
C       SYMMETRICAL SEMIANNUAL
      IF(WW(1).NE.9898) THEN
       WB(1,4)=(PB(17)*BT(3,1)+PB(31)*BT(5,1))*CD18
      ENDIF
      WB(2,4)=0
      WC(1,4)=0
      IF(WW(2).NE.9898) THEN
       WC(2,4)=-(PC(17)*BT(2,1)+PC(31)*BT(4,1))*CD18
      ENDIF
C       ASYMMETRICAL ANNUAL
      F5B=1.
      IF(WW(1).NE.9898) THEN
       WB(1,5)=(PB(10)*BT(2,1)+PB(11)*BT(4,1))*CD14*F5B
      ENDIF
      WB(2,5)=0
      F5C=1.
      WC(1,5)=0
      IF(WW(2).NE.9898) THEN
       WC(2,5)=-(PC(10)*BT(3,1)+PC(11)*BT(5,1))*CD14*F5C
      ENDIF
C       ASYMMETRICAL SEMIANNUAL
C       DIURNAL      
      IF(SW(7).EQ.0) GOTO 200
      F7B=1.
      F75B=1.
      IF(WW(1).NE.9898) THEN
       WB(1,7)=(PB(7)*BT(2,2)+PB(8)*BT(4,2)+PB(29)*BT(6,2)
     $         +PB(89)*BT(3,2)
     $  )*SSTL*F7B
     $ +(PB(13)*BT(3,2)+PB(146)*BT(5,2))
     $    *CD14*SSTL*F75B*SWC(5)
     $ + (PB(4)*BT(2,2)+PB(5)*BT(4,2)+PB(28)*BT(6,2)
     $         +PB(88)*BT(3,2)
     $  )*CSTL*F7B
     $ +(PB(12)*BT(3,2)+PB(145)*BT(5,2))
     $      *CD14*CSTL*F75B*SWC(5)
      ENDIF
      IF(WW(2).NE.9898) THEN
       WB(2,7)=-(PB(4)*BP(2,2)+PB(5)*BP(4,2)+PB(28)*BP(6,2)
     $         +PB(88)*BP(3,2)
     $  )*SSTL*F7B
     $ -(PB(12)*BP(3,2)+PB(145)*BP(5,2))
     $    *CD14*SSTL*F75B*SWC(5)
     $ + (PB(7)*BP(2,2)+PB(8)*BP(4,2)+PB(29)*BP(6,2)
     $         +PB(89)*BP(3,2)
     $  )*CSTL*F7B
     $ +(PB(13)*BP(3,2)+PB(146)*BP(5,2))
     $    *CD14*CSTL*F75B*SWC(5)
      ENDIF
      F7C=1.
      F75C=1.
      IF(WW(1).NE.9898) THEN
       WC(1,7)=-(PC(4)*BP(3,2)+PC(5)*BP(5,2)+PC(28)*BP(7,2)
     $         +PC(88)*BP(2,2)
     $   +PC(141)*BP(9,2)+PC(143)*BP(11,2)
     $  )*SSTL*F7C
     $ -(PC(12)*BP(2,2)+PC(145)*BP(4,2))
     $    *CD14*SSTL
     $   *F75C*SWC(5)
     $ +(PC(7)*BP(3,2)+PC(8)*BP(5,2)+PC(29)*BP(7,2)
     $         +PC(89)*BP(2,2)
     $ +PC(142)*BP(9,2)+PC(144)*BP(11,2)
     $  )*CSTL*F7C
     $ +(PC(13)*BP(2,2)+PC(146)*BP(4,2))
     $     *CD14*CSTL
     $   *F75C*SWC(5)
      ENDIF
      IF(WW(2).NE.9898) THEN
       WC(2,7)=-(PC(7)*BT(3,2)+PC(8)*BT(5,2)+PC(29)*BT(7,2)
     $         +PC(89)*BT(2,2)
     $ +PC(142)*BT(9,2)+PC(144)*BT(11,2)
     $  )*SSTL*F7C
     $ -(PC(13)*BT(2,2)+PC(146)*BT(4,2))
     $    *CD14*SSTL
     $   *F75C*SWC(5)
     $ -(PC(4)*BT(3,2)+PC(5)*BT(5,2)+PC(28)*BT(7,2)
     $         +PC(88)*BT(2,2)
     $ +PC(141)*BT(9,2)+PC(143)*BT(11,2)
     $  )*CSTL*F7C
     $ -(PC(12)*BT(2,2)+PC(145)*BT(4,2))
     $    *CD14*CSTL
     $   *F75C*SWC(5)
      ENDIF
  200 CONTINUE
C       SEMIDIURNAL
      IF(SW(8).EQ.0) GOTO 210
      F8B=1.+PB(90)*DFC*SWC(1)
      IF(WW(1).NE.9898) THEN
       WB(1,8)=(PB(9)*BT(3,3)+PB(43)*BT(5,3)+PB(111)*BT(7,3)
     $         +PB(98)*BT(4,3)
     $   +(PB(34)*BT(4,3)+PB(148)*BT(6,3))*CD14*SWC(5)
     $   +(PB(37)*BT(4,3))*CD19B*SWC(6)
     $  )*S2STL*F8B
     $ +(PB(6)*BT(3,3)+PB(42)*BT(5,3)+PB(110)*BT(7,3)
     $         +PB(96)*BT(4,3)
     $   +(PB(24)*BT(4,3)+PB(147)*BT(6,3))*CD14*SWC(5)
     $   +(PB(36)*BT(4,3))*CD19B*SWC(6)
     $  )*C2STL*F8B
      ENDIF
      IF(WW(2).NE.9898) THEN
       WB(2,8)=-(PB(6)*BP(3,3)+PB(42)*BP(5,3)+PB(110)*BP(7,3)
     $          +PB(96)*BP(4,3)
     $   +(PB(24)*BP(4,3)+PB(147)*BP(6,3))*CD14*SWC(5)
     $   +(PB(36)*BP(4,3))*CD19B*SWC(6)
     $  )*S2STL*F8B
     $   + (PB(9)*BP(3,3)+PB(43)*BP(5,3)+PB(111)*BP(7,3)
     $          +PB(98)*BP(4,3)
     $   +(PB(34)*BP(4,3)+PB(148)*BP(6,3))*CD14*SWC(5)
     $   +(PB(37)*BP(4,3))*CD19B*SWC(6)
     $  )*C2STL*F8B
      ENDIF
      F8C=1.+PC(90)*DFC*SWC(1)
      IF(WW(1).NE.9898) THEN
       WC(1,8)=-(PC(6)*BP(4,3)+PC(42)*BP(6,3)+PC(110)*BP(8,3)
     $          +PC(96)*BP(3,3)
     $   +(PC(24)*BP(3,3)+PC(147)*BP(5,3))*CD14*SWC(5)
     $   +(PC(36)*BP(3,3))*CD19B*SWC(6)
     $  )*S2STL*F8C
     $ +(PC(9)*BP(4,3)+PC(43)*BP(6,3)+PC(111)*BP(8,3)
     $          +PC(98)*BP(3,3)
     $   +(PC(34)*BP(3,3)+PC(148)*BP(5,3))*CD14*SWC(5)
     $   +(PC(37)*BP(3,3))*CD19B*SWC(6)
     $  )*C2STL*F8C
      ENDIF
      IF(WW(2).NE.9898) THEN
       WC(2,8)=-(PC(9)*BT(4,3)+PC(43)*BT(6,3)+PC(111)*BT(8,3)
     $          +PC(98)*BT(3,3)
     $   +(PC(34)*BT(3,3)+PC(148)*BT(5,3))*CD14*SWC(5)
     $   +(PC(37)*BT(3,3))*CD19B*SWC(6)
     $  )*S2STL*F8C
     $ - (PC(6)*BT(4,3)+PC(42)*BT(6,3)
     $          +PC(96)*BT(3,3)
     $   +PC(110)*BT(8,3)
     $   +(PC(24)*BT(3,3)+PC(147)*BT(5,3))*CD14*SWC(5)
     $   +(PC(36)*BT(3,3))*CD19B*SWC(6)
     $  )*C2STL*F8C
      ENDIF
  210 CONTINUE
C        TERDIURNAL
C        MAGNETIC ACTIVITY
      IF(SW(9).EQ.0) GOTO 40
      IF(SW9.EQ.-1.) GOTO 30
C           daily AP
      APD=AP(1)-4.
      APDF=(APD+(PB(45)-1.)*(APD+(EXP(-PB(44)*APD)-1.)/PB(44)))
C      APDFC=(APD+(PC(45)-1.)*(APD+(EXP(-PC(44)*APD)-1.)/PC(44)))
      APDFC=APDF
      IF(APD.EQ.0) GOTO 40
      IF(WW(1).NE.9898) THEN
       WB(1,9)=(PB(46)*BT(3,1)+PB(35)*BT(5,1))*APDF
     $    +(PB(122)*BT(2,2)+PB(123)*BT(4,2)+PB(124)*BT(6,2)
     $       )*COS(HR*(STL-PB(125)))*APDF*SWC(7)
      ENDIF
      IF(WW(2).NE.9898) THEN
       WB(2,9)=
     $   (PB(122)*BP(2,2)+PB(123)*BP(4,2)+PB(124)*BP(6,2)
     $     )*COS(HR*(STL-PB(125)+6.))*APDF*SWC(7)
      ENDIF
      IF(WW(1).NE.9898) THEN
       WC(1,9)=
     $   (PC(122)*BP(3,2)+PC(123)*BP(5,2)+PC(124)*BP(7,2)
     $       )*COS(HR*(STL-PC(125)))*APDFC*SWC(7)
      ENDIF
      IF(WW(2).NE.9898) THEN
       WC(2,9)=-(PC(46)*BT(2,1)+PC(35)*BT(4,1))*APDFC
     $  +(PC(122)*BT(3,2)+PC(123)*BT(5,2)+PC(124)*BT(7,2)
     $       )*COS(HR*(STL-PC(125)+6.))*APDFC*SWC(7)
      ENDIF
      GO TO 40
   30 CONTINUE
      IF(PB(25).LT.1.E-4) PB(25)=1.E-4
      APT=G0(AP(2))
      IF(APT.EQ.0) GOTO 40
      IF(WW(1).NE.9898) THEN
       WB(1,9)=(PB(97)*BT(3,1)+PB(55)*BT(5,1))*APT
     $    +(PB(129)*BT(2,2)+PB(130)*BT(4,2)+PB(131)*BT(6,2)
     $       )*COS(HR*(STL-PB(132)))*APT*SWC(7)
      ENDIF
      IF(WW(2).NE.9898) THEN
       WB(2,9)=
     $   (PB(129)*BP(2,2)+PB(130)*BP(4,2)+PB(131)*BP(6,2)
     $     )*COS(HR*(STL-PB(132)+6.))*APT*SWC(7)
      ENDIF
      IF(WW(1).NE.9898) THEN
       WC(1,9)=
     $   (PC(129)*BP(3,2)+PC(130)*BP(5,2)+PC(131)*BP(7,2)
     $       )*COS(HR*(STL-PC(132)))*APT*SWC(7)
      ENDIF
      IF(WW(2).NE.9898) THEN
       WC(2,9)=-(PC(97)*BT(2,1)+PC(55)*BT(4,1))*APT
     $  +(PC(129)*BT(3,2)+PC(130)*BT(5,2)+PC(131)*BT(7,2)
     $       )*COS(HR*(STL-PC(132)+6.))*APT*SWC(7)
      ENDIF
  40  CONTINUE
      WBT(1)=0
      WBT(2)=0
      WCT(1)=0
      WCT(2)=0
C       SUM WINDS AND CHANGE MERIDIONAL SIGN TO + NORTH
      DO 50 K=1,NSW
        WBT(1)=WBT(1)-ABS(SW(K))*WB(1,K)
        WCT(1)=WCT(1)-ABS(SW(K))*WC(1,K)
        WBT(2)=WBT(2)+ABS(SW(K))*WB(2,K)
        WCT(2)=WCT(2)+ABS(SW(K))*WC(2,K)
   50 CONTINUE
      IF(WW(1).NE.9898) WW(1)=WBT(1)*SW(24)+WCT(1)*SW(25)
      IF(WW(2).NE.9898) WW(2)=WBT(2)*SW(24)+WCT(2)*SW(25)
      RETURN
      END
C-----------------------------------------------------------------------
      SUBROUTINE GLBW5S(IYD,LAT,LONG,STL,PB,PC,WW)
      REAL LAT,LONG
      DIMENSION WB(2,15),WC(2,15),PB(1),PC(1),WW(2)
      COMMON/CSW/SW(25),ISW,SWC(25)
      COMMON/HWMC/WBT(2),WCT(2)
      COMMON/VPOLY2/XVL,LVL,MVL,CLAT,SLAT,BT(20,20),BP(20,20)
      COMMON/LTCOMP/TLL,NSVL,CSTL,SSTL,C2STL,S2STL,C3STL,S3STL
      COMMON/LGCOMP/XLL,NGVL,CLONG,SLONG,C2LONG,S2LONG
      SAVE
      DATA DGTR/.017453/,SR/7.2722E-5/,HR/.2618/,DR/1.72142E-2/
      DATA PB14/-1./,PB18/-1./,PC14/-1./,PC18/-1./,PSET/5./
      DATA NSW/14/,WB/30*0/,WC/30*0/
C       CONFIRM PARAMETER SET
      IF(PB(100).EQ.0) PB(100)=PSET
      IF(PB(100).NE.PSET) THEN
        WRITE(6,900) PSET,PB(100),PC(100)
  900   FORMAT(1X,'WRONG PARAMETER SET FOR GLBW5S',3F10.1)
        STOP
      ENDIF
C
      DO 10 J=1,NSW
        WB(1,J)=0
        WB(2,J)=0
        WC(1,J)=0
        WC(2,J)=0
   10 CONTINUE
      IYR = IYD/1000
      DAY = IYD - IYR*1000
C
      LV=7
      MV=2
      IF(XVL.NE.LAT.OR.LV.GT.LVL.OR.MV.GT.MVL) THEN
        SLAT=SIN(DGTR*LAT)
        CLAT=COS(DGTR*LAT)
        CALL VSPHR1(SLAT,CLAT,LV,MV,BT,BP,20)
        PLG10=SLAT
        PLG30=(5.*SLAT*SLAT-3.)*SLAT/2.
        XVL=LAT
        LVL=LV
        MVL=MV
      ENDIF
C
      NSV=2
      IF(TLL.NE.STL.OR.NSV.GT.NSVL)  THEN
        SSTL = SIN(HR*STL)
        CSTL = COS(HR*STL)
        S2STL = SIN(2.*HR*STL)
        C2STL = COS(2.*HR*STL)
        TLL = STL
        NSVL=NSV
      ENDIF
      IF(DAY.NE.DAYL.OR.PB(14).NE.PB14) CD14B=COS(DR*(DAY-PB(14)))
      IF(DAY.NE.DAYL.OR.PC(14).NE.PC14) CD14C=COS(DR*(DAY-PC(14)))
      IF(DAY.NE.DAYL.OR.PB(18).NE.PB18) CD18B=COS(2.*DR*(DAY-PB(18)))
      IF(DAY.NE.DAYL.OR.PC(18).NE.PC18) CD18C=COS(2.*DR*(DAY-PC(18)))
      IF(DAY.NE.DAYL.OR.PB(19).NE.PB19) CD19B=COS(2.*DR*(DAY-PB(19)))
      IF(DAY.NE.DAYL.OR.PB(25).NE.PB25) CD25B=COS(DR*(DAY-PB(25)))
C      IF(DAY.NE.DAYL.OR.PC(25).NE.PC25) CD25C=COS(DR*(DAY-PC(25)))
      IF(DAY.NE.DAYL.OR.PB(26).NE.PB26) CD26B=COS(DR*(DAY-PB(26)))
C      IF(DAY.NE.DAYL.OR.PC(26).NE.PC26) CD26C=COS(DR*(DAY-PC(26)))
      IF(DAY.NE.DAYL.OR.PC(32).NE.PC32) CD32C=COS(DR*(DAY-PC(32)))
      IF(DAY.NE.DAYL.OR.PC(39).NE.PC39) CD39C=COS(2.*DR*(DAY-PC(39)))
      IF(DAY.NE.DAYL.OR.PC(64).NE.PC64) CD64C=COS(DR*(DAY-PC(64)))
      IF(DAY.NE.DAYL.OR.PC(87).NE.PC87) CD87C=COS(2.*DR*(DAY-PC(87)))
      DAYL=DAY           
      PB14=PB(14)
      PC14=PC(14)
      PB18=PB(18)
      PC18=PC(18)
      PB19=PB(19)
      PB25=PB(25)
      PC25=PC(25)
      PB26=PB(26)
      PC26=PC(26)
      PC32=PC(32)
      PC39=PC(39)
      PC64=PC(64)
      PC87=PC(87)
C
      NGV=1
      IF(XLL.NE.LONG.OR.NGV.GT.NGVL) THEN
        SLONG=SIN(DGTR*LONG)
        CLONG=COS(DGTR*LONG)
        XLL=LONG
        NGVL=NGV
      ENDIF
C       TIME INDEPENDENT
      F1B=1.
      IF(WW(1).NE.9898) THEN
       WB(1,2)=(PB(2)*BT(3,1)+PB(3)*BT(5,1)+PB(23)*BT(7,1))*F1B
      ENDIF
      WB(2,2)=0.
      F1C=1.
      WC(1,2)=0.
      IF(WW(2).NE.9898) THEN
       WC(2,2)=-(PC(2)*BT(2,1)+PC(3)*BT(4,1)+PC(23)*BT(6,1))*F1C
     $ -(PC(27)*BT(3,1)+PC(15)*BT(5,1)+PC(60)*BT(7,1))*F1C
      ENDIF
C       SYMMETRICAL ANNUAL
      IF(WW(2).NE.9898) THEN
       WC(2,3)=-(PC(48)*BT(2,1)+PC(30)*BT(4,1))*CD32C
      ENDIF
C       SYMMETRICAL SEMIANNUAL
      IF(WW(1).NE.9898) THEN
       WB(1,4)=(PB(17)*BT(3,1)+PB(31)*BT(5,1))*CD18B
      ENDIF
      WB(2,4)=0
      WC(1,4)=0
      IF(WW(2).NE.9898) THEN
       WC(2,4)=-(PC(17)*BT(2,1)+PC(31)*BT(4,1)+PC(50)*BT(6,1))*CD18C
      ENDIF
C       ASYMMETRICAL ANNUAL
      F5B=1.
      IF(WW(1).NE.9898) THEN
       WB(1,5)=(PB(10)*BT(2,1)+PB(11)*BT(4,1))*CD14B*F5B
      ENDIF
      WB(2,5)=0
      F5C=1.
      WC(1,5)=0
      IF(WW(2).NE.9898) THEN
       WC(2,5)=-(PC(10)*BT(3,1)+PC(11)*BT(5,1)+PC(21)*BT(7,1))*CD14C*F5C
      ENDIF
C       ASYMMETRICAL SEMIANNUAL
      IF(WW(2).NE.9898) THEN
       WC(2,6)=-(PC(38)*BT(3,1)+PC(99)*BT(5,1))*CD39C
      ENDIF
C       DIURNAL      
      IF(SW(7).EQ.0) GOTO 200
      F7B=1.
      F75B=1.
      IF(WW(1).NE.9898) THEN
       WB(1,7)=(PB(7)*BT(2,2)+PB(8)*BT(4,2)+PB(29)*BT(6,2)
     $         +PB(89)*BT(3,2)
     $  )*SSTL*F7B
     $ +(PB(13)*BT(3,2))
     $    *CD25B*SSTL*F75B*SWC(5)
     $ + (PB(4)*BT(2,2)+PB(5)*BT(4,2)+PB(28)*BT(6,2)
     $         +PB(88)*BT(3,2)
     $  )*CSTL*F7B
     $ +(PB(12)*BT(3,2))
     $      *CD25B*CSTL*F75B*SWC(5)
      ENDIF
      IF(WW(2).NE.9898) THEN
       WB(2,7)=-(PB(4)*BP(2,2)+PB(5)*BP(4,2)+PB(28)*BP(6,2)
     $         +PB(88)*BP(3,2)
     $  )*SSTL*F7B
     $ -(PB(12)*BP(3,2))
     $    *CD25B*SSTL*F75B*SWC(5)
     $ + (PB(7)*BP(2,2)+PB(8)*BP(4,2)+PB(29)*BP(6,2)
     $         +PB(89)*BP(3,2)
     $  )*CSTL*F7B
     $ +(PB(13)*BP(3,2))
     $    *CD25B*CSTL*F75B*SWC(5)
      ENDIF
      F7C=1.
      F75C=1.
      IF(WW(1).NE.9898) THEN
       WC(1,7)=-(PC(4)*BP(3,2)+PC(5)*BP(5,2)+PC(28)*BP(7,2)
     $         +PC(88)*BP(2,2)
     $  )*SSTL*F7C
     $ -(PC(12)*BP(2,2))
     $    *CD25B*SSTL
     $   *F75C*SWC(5)
     $ +(PC(7)*BP(3,2)+PC(8)*BP(5,2)+PC(29)*BP(7,2)
     $         +PC(89)*BP(2,2)
     $  )*CSTL*F7C
     $ +(PC(13)*BP(2,2))
     $     *CD25B*CSTL
     $   *F75C*SWC(5)
      ENDIF
      IF(WW(2).NE.9898) THEN
       WC(2,7)=-(PC(7)*BT(3,2)+PC(8)*BT(5,2)+PC(29)*BT(7,2)
     $         +PC(89)*BT(2,2)
     $  )*SSTL*F7C
     $ -(PC(13)*BT(2,2))
     $    *CD25B*SSTL
     $   *F75C*SWC(5)
     $ -(PC(4)*BT(3,2)+PC(5)*BT(5,2)+PC(28)*BT(7,2)
     $         +PC(88)*BT(2,2)
     $  )*CSTL*F7C
     $ -(PC(12)*BT(2,2))
     $    *CD25B*CSTL
     $   *F75C*SWC(5)
      ENDIF
  200 CONTINUE
C       SEMIDIURNAL
      IF(SW(8).EQ.0) GOTO 210
      F8B=1.
      IF(WW(1).NE.9898) THEN
       WB(1,8)=(PB(9)*BT(3,3)+PB(43)*BT(5,3)+PB(35)*BT(7,3)
     $         +PB(98)*BT(4,3)
     $   +(PB(34)*BT(4,3))*CD26B*SWC(5)
     $   +(PB(37)*BT(4,3))*CD19B*SWC(6)
     $  )*S2STL*F8B
     $ +(PB(6)*BT(3,3)+PB(42)*BT(5,3)+PB(33)*BT(7,3)
     $         +PB(96)*BT(4,3)
     $   +(PB(24)*BT(4,3))*CD26B*SWC(5)
     $   +(PB(36)*BT(4,3))*CD19B*SWC(6)
     $  )*C2STL*F8B
      ENDIF
      IF(WW(2).NE.9898) THEN
       WB(2,8)=-(PB(6)*BP(3,3)+PB(42)*BP(5,3)+PB(33)*BP(7,3)
     $          +PB(96)*BP(4,3)
     $   +(PB(24)*BP(4,3))*CD26B*SWC(5)
     $   +(PB(36)*BP(4,3))*CD19B*SWC(6)
     $  )*S2STL*F8B
     $   + (PB(9)*BP(3,3)+PB(43)*BP(5,3)+PB(35)*BP(7,3)
     $          +PB(98)*BP(4,3)
     $   +(PB(34)*BP(4,3))*CD26B*SWC(5)
     $   +(PB(37)*BP(4,3))*CD19B*SWC(6)
     $  )*C2STL*F8B
      ENDIF
      F8C=1.
      IF(WW(1).NE.9898) THEN
       WC(1,8)=-(PC(6)*BP(4,3)+PC(42)*BP(6,3)+PC(33)*BP(8,3)
     $          +PC(96)*BP(3,3)
     $   +(PC(24)*BP(3,3))*CD26B*SWC(5)
     $   +(PC(36)*BP(3,3))*CD19B*SWC(6)
     $  )*S2STL*F8C
     $ +(PC(9)*BP(4,3)+PC(43)*BP(6,3)+PC(35)*BP(8,3)
     $          +PC(98)*BP(3,3)
     $   +(PC(34)*BP(3,3))*CD26B*SWC(5)
     $   +(PC(37)*BP(3,3))*CD19B*SWC(6)
     $  )*C2STL*F8C
      ENDIF
      IF(WW(2).NE.9898) THEN
       WC(2,8)=-(PC(9)*BT(4,3)+PC(43)*BT(6,3)+PC(35)*BT(8,3)
     $          +PC(98)*BT(3,3)
     $   +(PC(34)*BT(3,3))*CD26B*SWC(5)
     $   +(PC(37)*BT(3,3))*CD19B*SWC(6)
     $  )*S2STL*F8C
     $ - (PC(6)*BT(4,3)+PC(42)*BT(6,3)+PC(33)*BT(8,3)
     $          +PC(96)*BT(3,3)
     $   +(PC(24)*BT(3,3))*CD26B*SWC(5)
     $   +(PC(36)*BT(3,3))*CD19B*SWC(6)
     $  )*C2STL*F8C
      ENDIF
  210 CONTINUE
C        LONGITUDINAL
      IF(SW(10).EQ.0.OR.SW(11).EQ.0) GOTO 230
      IF(WW(1).NE.9898) THEN
       WC(1,11)=
     $ - (PC(65)*BP(2,2)+PC(66)*BP(4,2)+PC(67)*BP(6,2)
     $   +PC(75)*BP(3,2)+PC(76)*BP(5,2)+ PC(77)*BP(7,2)
     $   +(PC(57)*BP(2,2)+PC(59)*BP(4,2)+PC(62)*BP(6,2)
     $    +PC(51)*BP(3,2)+PC(53)*BP(5,2)+PC(55)*BP(7,2))
     $     *CD64C*SWC(3)
     $   +(PC(74)*BP(2,2)+PC(82)*BP(4,2)+PC(85)*BP(6,2)
     $    +PC(68)*BP(3,2)+PC(70)*BP(5,2)+PC(72)*BP(7,2))
     $     *CD87C*SWC(4)
     $  )*SLONG
     $ + (PC(91)*BP(2,2)+PC(92)*BP(4,2)+PC(93)*BP(6,2)
     $   +PC(78)*BP(3,2)+PC(79)*BP(5,2)+PC(80)*BP(7,2)
     $   +(PC(58)*BP(2,2)+PC(61)*BP(4,2)+PC(63)*BP(6,2)
     $    +PC(52)*BP(3,2)+PC(54)*BP(5,2)+PC(56)*BP(7,2))
     $     *CD64C*SWC(3)
     $   +(PC(81)*BP(2,2)+PC(84)*BP(4,2)+PC(86)*BP(6,2)
     $    +PC(69)*BP(3,2)+PC(71)*BP(5,2)+PC(73)*BP(7,2))
     $     *CD87C*SWC(4)
     $  )*CLONG
      ENDIF
      IF(WW(2).NE.9898) THEN
       WC(2,11)=
     $ - (PC(91)*BT(2,2)+PC(92)*BT(4,2)+PC(93)*BT(6,2)
     $   +PC(78)*BT(3,2)+PC(79)*BT(5,2)+PC(80)*BT(7,2)
     $   +(PC(58)*BT(2,2)+PC(61)*BT(4,2)+PC(63)*BT(6,2)
     $    +PC(52)*BT(3,2)+PC(54)*BT(5,2)+PC(56)*BT(7,2))
     $     *CD64C*SWC(3)
     $   +(PC(81)*BT(2,2)+PC(84)*BT(4,2)+PC(86)*BT(6,2)
     $    +PC(69)*BT(3,2)+PC(71)*BT(5,2)+PC(73)*BT(7,2))
     $     *CD87C*SWC(4)
     $  )*SLONG
     $ - (PC(65)*BT(2,2)+PC(66)*BT(4,2)+PC(67)*BT(6,2)
     $   +PC(75)*BT(3,2)+PC(76)*BT(5,2)+PC(77)*BT(7,2)
     $   +(PC(57)*BT(2,2)+PC(59)*BT(4,2)+PC(62)*BT(6,2)
     $    +PC(51)*BT(3,2)+PC(53)*BT(5,2)+PC(55)*BT(7,2))
     $     *CD64C*SWC(3)
     $   +(PC(74)*BT(2,2)+PC(82)*BT(4,2)+PC(85)*BT(6,2)
     $    +PC(68)*BT(3,2)+PC(70)*BT(5,2)+PC(72)*BT(7,2))
     $     *CD87C*SWC(4)
     $  )*CLONG
      ENDIF
  230 CONTINUE
      WBT(1)=0
      WBT(2)=0
      WCT(1)=0
      WCT(2)=0
C       SUM WINDS AND CHANGE MERIDIONAL SIGN TO + NORTH
      DO 50 K=1,NSW
        WBT(1)=WBT(1)-ABS(SW(K))*WB(1,K)
        WCT(1)=WCT(1)-ABS(SW(K))*WC(1,K)
        WBT(2)=WBT(2)+ABS(SW(K))*WB(2,K)
        WCT(2)=WCT(2)+ABS(SW(K))*WC(2,K)
   50 CONTINUE
      IF(WW(1).NE.9898) WW(1)=WBT(1)*SW(24)+WCT(1)*SW(25)
      IF(WW(2).NE.9898) WW(2)=WBT(2)*SW(24)+WCT(2)*SW(25)
      RETURN
      END
C-----------------------------------------------------------------------
      SUBROUTINE TSELEC(SV)
C        SET SWITCHES
C        SW FOR MAIN TERMS, SWC FOR CROSS TERMS
      DIMENSION SV(1),SAV(25),SVV(1)
      COMMON/CSW/SW(25),ISW,SWC(25)
      DO 100 I = 1,25
        SAV(I)=SV(I)
        SW(I)=AMOD(SV(I),2.)
        IF(ABS(SV(I)).EQ.1.OR.ABS(SV(I)).EQ.2.) THEN
          SWC(I)=1.
        ELSE
          SWC(I)=0.
        ENDIF
  100 CONTINUE
      ISW=64999
      RETURN
      ENTRY TRETRV(SVV)
      DO 200 I=1,25
        SVV(I)=SAV(I)
  200 CONTINUE
      END
C-----------------------------------------------------------------------
      SUBROUTINE VSPHR1(C,S,L,M,BT,BP,LMAX)
C      CALCULATE VECTOR SPHERICAL HARMONIC B FIELD THETA AND PHI
C      FUNCTIONS BT,BP THROUGH ORDER L,M FOR COLATITUDE (THETA)
C      WITH COSINE C AND SINE S OF COLATITUDE
C      BT(L+1,M+1)= [(L-M+1) P(L+1,M) - (L+1) P(L,M) COS(THETA)] /
C                [SQRT(L(L+1)) SIN(THETA)]
C      BP(L+1,M+1)= M P(L,M) /[SQRT(L(L+1)) SIN(THETA)]
C       RESULT FOR GIVEN L,M SAVED IN BT AND BP AT ONE HIGHER INDEX NUM
      DIMENSION BT(LMAX,1),BP(LMAX,1),PLG(20,20)
      SAVE
      DATA DGTR/1.74533E-2/
      IF(M.GT.L.OR.L.GT.LMAX-1) THEN
        WRITE(6,100) L,M,LMAX
  100   FORMAT('ILLEGAL INDICES TO VSPHER',3I6)
        RETURN
      ENDIF
      BT(1,1)=0
      BP(1,1)=0
      IF(L.EQ.0.AND.M.EQ.0) RETURN
      CALL LEGPL1(C,S,L+1,M,PLG,20)
      IF(ABS(S).LT.1.E-5) THEN
        IC=SIGN(1.,S)
        S=0
      ENDIF
      DO 20 LL=1,L
        SQT=SQRT(FLOAT(LL)*(FLOAT(LL)+1))
        LMX=MIN(LL,M)
        DO 15 MM=0,LMX
          IF(S.EQ.0) THEN
            IF(MM.NE.1) THEN
              BT(LL+1,MM+1)=0
              BP(LL+1,MM+1)=0
            ELSE
              BT(LL+1,MM+1)=(LL*(LL+1)*(LL+2)*.5*(IC)**(LL+2)
     $           -(LL+1)*C*LL*(LL+1)*.5*(IC)**(LL+1))/SQT
              BP(LL+1,MM+1)=MM*LL*(LL+1)*.5*(IC)**(LL+1)/SQT
            ENDIF
          ELSE
            BT(LL+1,MM+1)=((LL-MM+1)*PLG(LL+2,MM+1)
     $      -(LL+1)*C*PLG(LL+1,MM+1))/(S*SQT)
            BP(LL+1,MM+1)=MM*PLG(LL+1,MM+1)/(S*SQT)
          ENDIF
   15   CONTINUE
   20 CONTINUE
      END
C-----------------------------------------------------------------------
      SUBROUTINE LEGPL1(C,S,L,M,PLG,LMAX)
C      CALCULATE LEGENDRE POLYNOMIALS PLG(L+1,M+1) THROUGH ORDER L,M 
C      FOR COSINE C AND SINE S OF COLATITUDE
      DIMENSION PLG(LMAX,1)
      SAVE
      DATA DGTR/1.74533E-2/
      IF(M.GT.L.OR.L.GT.LMAX-1) THEN
        WRITE(6,99) L,M,LMAX
   99 FORMAT(1X,'ILLEGAL INDICES TO LEGPOL',3I5)
        RETURN
      ENDIF
      PLG(1,1)=1.
      IF(L.EQ.0.AND.M.EQ.0) RETURN
C      CALCULATE L=M CASE AND L=M+1
      DO 10 MM=0,M
        IF(MM.GT.0) PLG(MM+1,MM+1)=PLG(MM,MM)*(2.*MM-1.)*S
        IF(L.GT.MM) PLG(MM+2,MM+1)=PLG(MM+1,MM+1)*(2.*MM+1)*C
   10 CONTINUE
      IF(L.EQ.1) RETURN
      MMX=MIN(M,L-2)
      DO 30 MM=0,MMX
        DO 20 LL=MM+2,L
          PLG(LL+1,MM+1)=((2.*LL-1.)*C*PLG(LL,MM+1)-
     $     (LL+MM-1.)*PLG(LL-1,MM+1))/(LL-MM)
   20   CONTINUE
   30 CONTINUE
      RETURN
      END
C-----------------------------------------------------------------------
      SUBROUTINE SPLINE(X,Y,N,YP1,YPN,Y2)
C        CALCULATE 2ND DERIVATIVES OF CUBIC SPLINE INTERP FUNCTION
C        X,Y: ARRAYS OF TABULATED FUNCTION IN ASCENDING ORDER BY X
C        N: SIZE OF ARRAYS X,Y
C        YP1,YPN: SPECIFIED DERIVATIVES AT X(1) AND X(N); VALUES
C                 >= 1E30 SIGNAL SIGNAL SECOND DERIVATIVE ZERO
C        Y2: OUTPUT ARRAY OF SECOND DERIVATIVES
      PARAMETER (NMAX=100)
      DIMENSION X(N),Y(N),Y2(N),U(NMAX)
      SAVE
      IF(YP1.GT..99E30) THEN
        Y2(1)=0
        U(1)=0
      ELSE
        Y2(1)=-.5
        U(1)=(3./(X(2)-X(1)))*((Y(2)-Y(1))/(X(2)-X(1))-YP1)
      ENDIF
      DO 11 I=2,N-1
        SIG=(X(I)-X(I-1))/(X(I+1)-X(I-1))
        P=SIG*Y2(I-1)+2.
        Y2(I)=(SIG-1.)/P
        U(I)=(6.*((Y(I+1)-Y(I))/(X(I+1)-X(I))-(Y(I)-Y(I-1))
     $    /(X(I)-X(I-1)))/(X(I+1)-X(I-1))-SIG*U(I-1))/P
   11 CONTINUE
      IF(YPN.GT..99E30) THEN
        QN=0
        UN=0
      ELSE
        QN=.5
        UN=(3./(X(N)-X(N-1)))*(YPN-(Y(N)-Y(N-1))/(X(N)-X(N-1)))
      ENDIF
      Y2(N)=(UN-QN*U(N-1))/(QN*Y2(N-1)+1.)
      DO 12 K=N-1,1,-1
        Y2(K)=Y2(K)*Y2(K+1)+U(K)
   12 CONTINUE
      RETURN
      END
C-----------------------------------------------------------------------
      SUBROUTINE SPLINT(XA,YA,Y2A,N,X,Y)
C        CALCULATE CUBIC SPLINE INTERP VALUE
C        XA,YA: ARRAYS OF TABULATED FUNCTION IN ASCENDING ORDER BY X
C        Y2A: ARRAY OF SECOND DERIVATIVES
C        N: SIZE OF ARRAYS XA,YA,Y2A
C        X: ABSCISSA FOR INTERPOLATION
C        Y: OUTPUT VALUE
      DIMENSION XA(N),YA(N),Y2A(N)
      SAVE
      KLO=1
      KHI=N
    1 CONTINUE
      IF(KHI-KLO.GT.1) THEN
        K=(KHI+KLO)/2
        IF(XA(K).GT.X) THEN
          KHI=K
        ELSE
          KLO=K
        ENDIF
        GOTO 1
      ENDIF
      H=XA(KHI)-XA(KLO)
      IF(H.EQ.0) WRITE(6,*) 'BAD XA INPUT TO SPLINT'
      A=(XA(KHI)-X)/H
      B=(X-XA(KLO))/H
      Y=A*YA(KLO)+B*YA(KHI)+
     $  ((A*A*A-A)*Y2A(KLO)+(B*B*B-B)*Y2A(KHI))*H*H/6.
      RETURN
      END
C-----------------------------------------------------------------------
      BLOCK DATA INITW5
C       For wind model GWS
      COMMON/CSW/SW(25),ISW,SWC(25)
      COMMON/VPOLY2/XVL,LVL,MVL,CLAT,SLAT,BT(20,20),BP(20,20)
      COMMON/LTCOMP/TLL,NSVL,CSTL,SSTL,C2STL,S2STL,C3STL,S3STL
      COMMON/LGCOMP/XLL,NGVL,CLONG,SLONG,C2LONG,S2LONG
      DATA ISW/0/
      DATA XVL/-999./,LVL/-1/,MVL/-1/
      DATA TLL/-999./,NSVL/-1/
      DATA XLL/-999./,NGVL/-1/
      END
C-----------------------------------------------------------------------
      BLOCK DATA GWSBK5
C          HWM93    28-JAN-93   
      COMMON/PARMW5/PBA1(50),PBA2(50),PBA3(50),PBA4(50),
     $PCA1(50),PCA2(50),PCA3(50),PCA4(50),
     $PBB1(50),PBB2(50),PBB3(50),PCB1(50),PCB2(50),PCB3(50),
     $PBC1(50),PBC2(50),PBC3(50),PCC1(50),PCC2(50),PCC3(50),
     $PBD1(50),PBD2(50),PBD3(50),PCD1(50),PCD2(50),PCD3(50),
     $PBE1(50),PBE2(50),PBE3(50),PCE1(50),PCE2(50),PCE3(50),
     $PBF1(50),PBF2(50),PBF3(50),PCF1(50),PCF2(50),PCF3(50),
     $PBG1(50),PBG2(50),PBG3(50),PCG1(50),PCG2(50),PCG3(50),
     $PBH1(50),PBH2(50),PBH3(50),PCH1(50),PCH2(50),PCH3(50),
     $PBI1(50),PBI2(50),PCI1(50),PCI2(50),PBJ1(50),PBJ2(50),
     $PCJ1(50),PCJ2(50),PBK1(50),PBK2(50),PCK1(50),PCK2(50),
     $PBL1(50),PBL2(50),PCL1(50),PCL2(50),PBM1(50),PBM2(50),
     $PCM1(50),PCM2(50),PBN1(50),PBN2(50),PCN1(50),PCN2(50),
     $PBO1(50),PBO2(50),PCO1(50),PCO2(50),PBP1(50),PBP2(50),
     $PCP1(50),PCP2(50),PBQ1(50),PBQ2(50),PCQ1(50),PCQ2(50),
     $PBR1(50),PBR2(50),PCR1(50),PCR2(50),PBS1(50),PBS2(50),
     $PCS1(50),PCS2(50),PBT1(50),PBT2(50),PCT1(50),PCT2(50),
     $PBU1(50),PBU2(50),PCU1(50),PCU2(50)
      COMMON/DATW/ISDATE(3),ISTIME(2),NAME(2)
      CHARACTER*4 :: ISDATE,ISTIME,NAME
      DATA ISDATE/'28-J','AN-9','3   '/,ISTIME/'20:3','5:39'/
      DATA NAME/'HWM9','3   '/
C         WINF
      DATA PBA1/
     *  0.00000E+00,-1.31640E+01,-1.52352E+01, 1.00718E+02, 3.94962E+00,
     *  2.19452E-01, 8.03296E+01,-1.02032E+00,-2.02149E-01, 5.67263E+01,
     *  0.00000E+00,-6.05459E+00, 6.68106E+00,-8.49486E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 8.39399E+01, 0.00000E+00, 9.96285E-02,
     *  0.00000E+00,-2.66243E-02, 0.00000E+00,-1.32373E+00, 1.39396E-02,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 3.36523E+01,-7.42795E-01,-3.89352E+00,-7.81354E-01,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 3.76631E+00,-1.22024E+00,
     * -5.47580E-01, 1.09146E+00, 9.06245E-01, 2.21119E-02, 0.00000E+00,
     *  7.73919E-02, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00/
      DATA PBA2/
     * -3.82415E-01, 0.00000E+00, 1.76202E-01, 0.00000E+00,-6.77651E-01,
     *  1.10357E+00, 2.25732E+00, 0.00000E+00, 1.54237E+04, 0.00000E+00,
     *  1.27411E-01,-2.84314E-03, 4.62562E-01,-5.34596E+01,-7.23808E+00,
     *  0.00000E+00, 0.00000E+00, 4.52770E-01,-8.50922E+00,-2.85389E-01,
     *  2.12000E+01, 6.80171E+02, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     * -2.72552E+04, 0.00000E+00, 0.00000E+00, 0.00000E+00, 2.64109E+03,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00,-1.47320E+00,-2.98179E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 1.05412E-02,
     *  4.93452E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 7.98332E-02,-5.30954E+01, 2.10211E-02, 3.00000E+00/
      DATA PBA3/
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,-2.79843E-01,
     *  1.81152E-01, 0.00000E+00, 0.00000E+00,-6.24673E-02,-5.37589E-02,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00,-8.94418E-02, 3.70413E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,-4.84645E+00,
     *  4.24178E-01, 0.00000E+00, 0.00000E+00, 1.86494E-01,-9.56931E-02/
      DATA PBA4/
     *  2.08426E+00, 1.53714E+00,-2.87496E-01, 4.06380E-01,-3.59788E-01,
     * -1.87814E-01, 0.00000E+00, 0.00000E+00, 2.01362E-01,-1.21604E-01,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 7.86304E+00,
     *  2.51878E+00, 2.91455E+00, 4.32308E+00, 6.77054E-02,-2.39125E-01,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  1.57976E+00,-5.44598E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     * -5.30593E-01,-5.02237E-01,-2.05258E-01, 2.62263E-01,-2.50195E-01,
     *  4.28151E-01, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00/
C         WINF
      DATA PCA1/
     *  0.00000E+00, 1.31026E+01,-4.93171E+01, 2.51045E+01,-1.30531E+01,
     *  6.56421E-01, 2.75633E+01, 4.36433E+00, 1.04638E+00, 5.77365E+01,
     *  0.00000E+00,-6.27766E+00, 2.33010E+00,-1.41351E+01, 2.49653E-01,
     *  0.00000E+00, 0.00000E+00, 8.00000E+01, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 1.03817E-02,-1.70950E+01,-1.92295E+00, 0.00000E+00,
     *  0.00000E+00,-1.17490E+01,-7.14788E-01, 6.72649E+00, 0.00000E+00,
     *  0.00000E+00,-1.57793E+02,-1.70815E+00,-7.92416E+00,-1.67372E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 1.87973E-01,
     * -1.61602E-01,-1.13832E-01,-7.22447E-01, 2.21119E-02, 0.00000E+00,
     * -3.01967E+00,-1.72798E-01,-5.15055E-03,-1.23477E-02, 3.60805E-03/
      DATA PCA2/
     * -1.36730E+00, 0.00000E+00, 1.24390E-02, 0.00000E+00,-1.36577E+00,
     *  3.18101E-02, 0.00000E+00, 0.00000E+00, 0.00000E+00,-1.39334E+01,
     *  1.42088E-02, 0.00000E+00, 0.00000E+00, 0.00000E+00,-4.72219E+00,
     * -7.47970E+00,-4.96528E+00, 0.00000E+00, 1.24712E+00,-2.56833E+01,
     * -4.26630E+01, 3.92431E+04,-2.57155E+00,-4.35589E-02, 0.00000E+00,
     *  0.00000E+00, 2.02425E+00,-1.48131E+00,-7.72242E-01, 2.99008E+04,
     *  4.50148E-03, 5.29718E-03,-1.26697E-02, 3.20909E-02, 0.00000E+00,
     *  0.00000E+00, 7.01739E+00, 3.11204E+00, 0.00000E+00, 0.00000E+00,
     * -2.13088E+00, 1.32789E+01, 5.07958E+00, 7.26537E-02, 2.87495E-01,
     *  9.97311E-03,-2.56440E+00, 0.00000E+00, 0.00000E+00, 3.00000E+00/
      DATA PCA3/
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00,-9.90073E-03,-3.27333E-02,
     * -4.30379E+01,-2.87643E+01,-5.91793E+00,-1.50460E+02, 0.00000E+00,
     *  0.00000E+00, 6.55038E-03, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 6.18051E-03, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  1.40484E+00, 5.54554E+00, 0.00000E+00, 0.00000E+00, 7.93810E+00,
     *  1.57192E+00, 1.03971E+00, 9.88279E-01,-4.37662E-02,-2.15763E-02/
      DATA PCA4/
     * -2.31583E+00, 4.32633E+00,-1.12716E+00, 3.38459E-01, 4.66956E-01,
     *  7.18403E-01, 5.80836E-02, 4.12653E-01, 1.04111E-01,-8.30672E-02,
     * -5.55541E+00,-4.97473E+00,-2.03007E+01, 0.00000E+00,-6.06235E-01,
     * -1.73121E-01, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 9.29850E-02,-6.38131E-02,
     *  3.93037E-02, 5.21942E-02, 2.26578E-02, 4.13157E-02, 0.00000E+00,
     *  6.28524E+00, 4.43721E+00,-4.31270E+00, 2.32787E+00, 2.55591E-01,
     *  1.60098E+00,-1.20649E+00, 3.05042E+00,-1.88944E+00, 5.35561E+00,
     *  2.02391E-01, 4.62950E-02, 3.39155E-01, 7.94007E-02, 6.30345E-01,
     *  1.93554E-01, 3.93238E-01, 1.76254E-01,-2.51359E-01,-7.06879E-01/
C       UGN1(1)
      DATA PBB1/
     *  6.22831E-01, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  5.90566E+00, 0.00000E+00, 0.00000E+00,-3.20571E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00,-8.30368E-01, 1.39396E-02,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 2.40657E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00,-4.80790E+00,-1.62744E+00, 2.21119E-02, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00/
      DATA PBB2/
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 4.00000E+00/
      DATA PBB3/
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 2.10531E-01,
     * -8.94829E-01, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00/
C         UGN1(1)
      DATA PCB1/
     *  5.45009E-01, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     * -3.60304E+00, 0.00000E+00, 0.00000E+00,-5.04071E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 5.62113E-01, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 1.14657E+01, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 4.65483E-01, 1.73636E+00, 2.21119E-02, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00/
      DATA PCB2/
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 4.00000E+00/
      DATA PCB3/
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,-8.30769E-01,
     *  7.73649E-01, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00/
C         UN1(1)
      DATA PBC1/
     *  6.09940E-01, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 1.39396E-02,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 2.21119E-02, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00/
      DATA PBC2/
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 4.00000E+00/
      DATA PBC3/
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00/
C        UN1(1)
      DATA PCC1/
     *  5.46739E-01, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 2.21119E-02, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00/
      DATA PCC2/
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 4.00000E+00/
      DATA PCC3/
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00/
C        UN1(2)
      DATA PBD1/
     *  4.99007E-01, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  2.59994E+00, 0.00000E+00, 0.00000E+00,-1.78418E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00,-5.24986E+00, 1.39396E-02,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 2.77918E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 2.21119E-02, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00/
      DATA PBD2/
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 4.00000E+00/
      DATA PBD3/
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 5.68996E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00/
C         UN1(2)
      DATA PCD1/
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     * -7.26156E+00, 0.00000E+00, 0.00000E+00,-4.12416E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00,-2.88934E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 3.65720E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 2.21119E-02, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00/
      DATA PCD2/
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 4.00000E+00/
      DATA PCD3/
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 2.01835E-01,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00/
C        UN1(3)
      DATA PBE1/
     *  0.00000E+00,-1.37217E+01, 0.00000E+00, 2.38712E-01,-3.92230E+00,
     *  6.11035E+00,-1.57794E+00,-5.87709E-01, 1.21178E+01, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 5.23202E+01, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00,-2.22836E+03, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00,-3.94006E+00, 1.39396E-02,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 3.99844E-01, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00,-1.38936E+00, 2.22534E+00, 2.21119E-02, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00/
      DATA PBE2/
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 4.00000E+00/
      DATA PBE3/
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 4.35518E-01, 8.40051E-01, 0.00000E+00,-8.88181E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 6.81729E-01, 9.67536E-01,
     *  0.00000E+00,-9.67836E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00/
C          UN1(3)
      DATA PCE1/
     *  0.00000E+00,-2.75655E+01,-6.61134E+00, 4.85118E+00, 8.15375E-01,
     * -2.62856E+00, 2.99508E-02,-2.00532E-01,-9.35618E+00, 1.17196E+01,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00,-2.43848E+00, 1.90065E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00,-3.37525E-01, 1.76471E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00/
      DATA PCE2/
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 4.00000E+00/
      DATA PCE3/
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00,-9.23682E-01,-8.84150E-02, 0.00000E+00,-9.88578E-01,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00,-1.00747E+00,-1.07468E-02,
     *  0.00000E+00,-3.66376E-01, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00/
C           UN1(4)
      DATA PBF1/
     *  0.00000E+00, 1.02709E+01, 0.00000E+00,-1.42016E+00,-4.90438E+00,
     * -9.11544E+00,-3.80570E+00,-2.09013E+00, 1.32939E+01,-1.28062E+01,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 1.23024E+01, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 3.92126E+02, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 1.39396E-02,
     *  0.00000E+00, 0.00000E+00,-5.56532E+00,-1.27046E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00,-3.03553E+00,-9.09832E-01, 2.21119E-02, 0.00000E+00,
     *  8.89965E-01, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00/
      DATA PBF2/
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 9.19210E-01, 0.00000E+00, 0.00000E+00, 4.00000E+00/
      DATA PBF3/
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00,-2.46693E-01, 7.44650E-02, 3.84661E-01, 9.44052E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00,-2.25083E-01, 1.54206E-01,
     *  4.41303E-01, 8.74742E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00/
C           UN1(4)
      DATA PCF1/
     *  0.00000E+00, 3.61143E+00,-8.24679E+00, 1.70751E+00, 1.16676E+00,
     *  6.24821E+00,-5.68968E-01, 8.53046E-01,-6.94168E+00, 1.04152E+01,
     * -3.70861E+01, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00,-1.23336E+01, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 5.33958E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00,-6.43682E-01,-1.00000E+00, 0.00000E+00,
     *  0.00000E+00,-1.00000E+00, 0.00000E+00,-5.47300E-01, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00,-8.58764E-01, 4.72310E-01, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00/
      DATA PCF2/
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 4.00000E+00/
      DATA PCF3/
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 3.37325E-01,-3.57698E-02,-6.97393E-01, 1.35387E+01,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 2.78162E-01,-2.33383E-01,
     * -7.12994E-01, 1.29234E+01, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00/
C         UN1(5)
      DATA PBG1/
     *  0.00000E+00,-1.71856E+00, 5.32877E+00, 5.33548E-01,-2.66034E+00,
     *  6.76192E-01, 2.25618E+00,-5.78954E-01,-2.69685E+00, 1.21933E+00,
     * -6.13650E+00, 7.79531E-01, 1.63652E+00, 3.63835E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 7.51539E+00,-5.27337E-01, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 1.06625E-01, 1.39396E-02,
     *  0.00000E+00, 0.00000E+00,-1.07240E+00,-8.31257E-01, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 7.04016E-01, 0.00000E+00,
     *  7.56158E-01,-4.21268E-02, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 1.02843E+00, 5.21034E-01, 2.21119E-02, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00/
      DATA PBG2/
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 4.12504E+00, 1.08459E-01, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     * -3.16261E-01, 0.00000E+00,-1.44288E-01, 0.00000E+00, 4.00000E+00/
      DATA PBG3/
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,-2.36181E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00/
C         UN1(5)
      DATA PCG1/
     *  0.00000E+00, 3.47155E+00, 1.76102E+01, 2.80371E+00,-2.08556E+00,
     *  1.10473E+00, 6.74582E+00,-5.75988E-01, 1.02708E+00,-2.23381E+01,
     *  8.60171E+00, 5.12046E-01,-8.12861E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 9.11036E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 3.89742E+00, 2.01725E-01, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 5.06308E-01, 2.04596E-01, 0.00000E+00,
     *  4.40377E+00, 0.00000E+00, 0.00000E+00, 2.20760E+00, 0.00000E+00,
     * -1.36478E+00, 2.38097E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00,-7.08949E-02,-1.61277E-01, 2.21119E-02, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00/
      DATA PCG2/
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00,-2.16898E+00,-5.31596E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  2.53060E+00, 0.00000E+00,-7.17287E-01, 0.00000E+00, 4.00000E+00/
      DATA PCG3/
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,-1.91762E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00/
C          UGN1(2 
      DATA PBH1/
     *  0.00000E+00,-7.70936E-01, 1.58158E+00, 3.61790E+00,-1.51748E+00,
     * -5.66098E-01, 1.69393E+00,-4.60489E-01,-8.31527E-01,-4.66437E-01,
     * -1.21750E+00, 0.00000E+00, 0.00000E+00, 1.56505E+02, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00,-5.19321E+01, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 1.39396E-02,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 3.09223E-01, 1.33715E-01, 2.21119E-02, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00/
      DATA PBH2/
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 4.00000E+00/
      DATA PBH3/
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00/
C          UGN1(2)
      DATA PCH1/
     *  0.00000E+00, 1.72324E-01, 3.08033E-01, 4.55771E-01, 1.46516E-01,
     *  1.97176E-01,-1.53329E-01, 6.91877E-02,-3.07184E-01, 2.65686E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00,-2.24369E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 4.04079E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  4.99627E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00,-7.83317E-03,-6.88967E-02, 2.21119E-02, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00/
      DATA PCH2/
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 4.00000E+00/
      DATA PCH3/
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00/
C          UN2(2)
      DATA PBI1/
     *  0.00000E+00,-7.99767E-01,-3.24774E-01, 7.70975E-01, 6.71796E-01,
     *  5.65483E-01,-2.99727E+00, 3.32448E+00,-9.15018E-01, 5.97656E+00,
     *  0.00000E+00,-1.19515E+00,-8.30457E-01, 3.26074E+00, 0.00000E+00,
     *  0.00000E+00,-1.58365E+00, 7.44825E-02, 5.91372E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00,-1.41511E-01,-3.01048E+00,
     *  2.35960E+01, 0.00000E+00,-1.70352E+00,-2.39746E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 1.30488E+00, 0.00000E+00,
     *  5.95132E-01, 5.64301E-01, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 5.30317E-01, 5.66569E-01, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00/
      DATA PBI2/
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 5.72367E+00, 1.58411E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  1.04557E-01, 0.00000E+00,-2.04710E-01, 0.00000E+00, 5.00000E+00/
C          UN2(2)
      DATA PCI1/
     *  0.00000E+00, 6.34487E+00, 9.84162E+00, 3.42136E+00,-5.10607E+00,
     * -8.58745E-02, 3.11501E+00, 5.34570E-01, 1.18027E+00, 4.28027E+00,
     *  4.75123E+00, 6.40947E-01,-4.15165E+00,-1.38154E+01, 0.00000E+00,
     *  0.00000E+00, 1.13145E+01,-5.15954E+00, 0.00000E+00, 0.00000E+00,
     *  1.35576E+01, 0.00000E+00,-5.78982E+00,-2.22043E+00, 3.36776E+00,
     *  3.04791E+01, 0.00000E+00, 2.94709E+00,-4.17536E-01,-1.59855E+00,
     * -2.18320E+00, 1.68269E+01, 0.00000E+00, 1.00829E+00, 0.00000E+00,
     * -6.85096E-01, 2.07822E-01, 3.50168E-01,-3.03662E+01, 0.00000E+00,
     *  0.00000E+00,-1.65726E-01,-8.97831E-02, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00,-5.24159E+00, 0.00000E+00,-3.52218E+00/
      DATA PCI2/
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 5.69093E-01,-7.44918E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  2.10865E+00, 0.00000E+00, 1.76776E-01, 1.54755E+00, 5.00000E+00/
C          UN2(3)
      DATA PBJ1/
     *  0.00000E+00, 2.28657E+00, 4.96548E-01, 6.99915E+00,-2.31540E+00,
     * -1.82163E-01,-5.00779E-01, 3.18199E-01,-6.14645E-01, 6.34816E+00,
     *  0.00000E+00, 7.94635E-01,-5.55565E-01, 3.85494E+00, 0.00000E+00,
     *  0.00000E+00,-3.96109E+00, 1.90775E-01, 4.51396E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00,-5.04618E-01,-4.14385E+00,
     *  2.30244E+01, 0.00000E+00, 1.00689E+00, 5.75680E-02, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 8.56741E-01, 0.00000E+00,
     *  9.54921E-02, 5.56659E-01, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 1.38503E-01, 4.50415E-01, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00/
      DATA PBJ2/
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 2.22813E-01,-8.63549E-02, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  1.37970E-01, 0.00000E+00,-3.25612E-01, 0.00000E+00, 5.00000E+00/
C          UN2(3)
      DATA PCJ1/
     *  0.00000E+00, 5.07608E+00, 3.31479E+00, 3.01548E-01,-1.12100E+00,
     * -7.63711E-02, 2.29748E+00,-1.36699E+00, 7.53433E-01, 3.60702E+01,
     * -1.55266E+00, 1.47382E+00,-2.53895E+00,-1.47720E+01, 0.00000E+00,
     *  0.00000E+00, 1.11787E+01,-1.06256E+01, 0.00000E+00, 0.00000E+00,
     *  7.86391E+00, 0.00000E+00,-8.61020E+00,-1.59313E+00,-5.17013E+00,
     *  1.20468E+00, 0.00000E+00, 5.76509E-01, 9.96195E-01,-1.45539E+00,
     * -1.79950E+01, 8.76957E+00, 0.00000E+00,-1.22863E+00, 0.00000E+00,
     * -6.19019E-01,-1.09571E-01,-4.31325E-02,-4.21981E+01, 0.00000E+00,
     *  0.00000E+00,-1.51519E-01,-1.24067E-01, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00,-6.39248E+00, 0.00000E+00, 6.64508E-01/
      DATA PCJ2/
     * -7.33184E-01,-9.72031E-03, 1.36789E+00,-8.62311E-01,-3.06395E-03,
     *  2.53354E-01,-2.40918E-01,-4.06932E-02,-5.82223E-01, 0.00000E+00,
     * -8.70285E-01, 7.72318E-01,-6.54213E-01,-2.19231E+01,-1.56509E-01,
     *  2.71745E-01, 5.93538E-01, 2.27757E-01,-5.98215E-01, 3.96457E-01,
     *  2.98705E-01, 1.78618E-01,-5.24538E-01, 1.16439E-01, 7.56829E-02,
     * -4.26809E-01, 5.77187E-01, 8.65450E-01,-7.53614E-01, 1.38381E-01,
     * -1.82265E-01, 2.85263E-01, 4.51322E-01, 1.02775E-01, 3.55731E-01,
     * -4.60896E-01,-3.13037E+01,-2.70818E+00,-7.84847E-01, 0.00000E+00,
     * -1.03473E-01,-3.87649E-01,-1.22613E-01, 0.00000E+00, 0.00000E+00,
     *  8.91325E-01, 0.00000E+00, 1.06189E-01, 9.13030E-02, 5.00000E+00/
C          UN2(4)
      DATA PBK1/
     *  0.00000E+00, 2.94921E+00, 2.79238E+00, 2.58949E+00, 3.56459E-01,
     *  3.12952E-01, 3.34337E+00,-2.83209E+00,-1.05979E+00, 3.92313E+00,
     *  0.00000E+00, 1.73703E-01,-3.23441E-01, 4.15836E+00, 0.00000E+00,
     *  0.00000E+00,-1.77156E+00, 6.44113E-01, 1.88743E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00,-4.64778E-01,-4.23560E+00,
     *  2.27271E+01, 0.00000E+00,-4.89468E-01, 1.82689E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 4.38217E-02, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 8.62449E-02, 4.46041E-01, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00/
      DATA PBK2/
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00,-1.40393E-01, 1.01821E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 5.00000E+00/
C          UN2(4)
      DATA PCK1/
     *  0.00000E+00, 6.04465E+00, 4.50924E+00, 3.84425E-02,-8.70772E-01,
     * -9.55408E-02, 2.28287E+00,-4.37834E-01, 3.57839E-01, 7.20721E+01,
     * -4.41757E+00,-9.13648E-01,-8.71866E-01,-6.26173E+00, 0.00000E+00,
     *  0.00000E+00, 5.92817E+00, 6.15853E+00, 0.00000E+00, 0.00000E+00,
     * -4.89060E+00, 0.00000E+00,-8.30378E+00, 1.07462E-01, 1.08471E+02,
     *  3.39150E+01,-4.57863E+00,-7.18349E-02,-2.71703E-01,-8.96297E+00,
     * -2.37986E+01, 4.11880E+00, 0.00000E+00,-9.95820E-01, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00,-8.91622E+00,-6.85950E+01, 0.00000E+00,
     *  0.00000E+00,-3.62769E-02,-1.65893E-01, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00,-2.94563E+00, 0.00000E+00, 1.23581E+00/
      DATA PCK2/
     * -6.06026E-01,-6.50229E-01, 1.91330E+00,-1.00314E+00, 1.13346E-01,
     *  4.21885E-01,-3.97688E-01,-2.77437E-01,-6.65893E-01, 0.00000E+00,
     * -1.37646E+00, 1.35171E+00,-9.55595E-01,-1.96450E+01,-2.50039E-01,
     *  5.93389E-01, 9.87131E-01, 5.43559E-01,-1.04322E+00, 6.32546E-01,
     *  3.73259E-01, 5.22657E-01,-5.81400E-01,-1.26425E-01,-1.29843E-01,
     * -5.36598E-01, 8.02402E-01, 9.04347E-01,-1.10799E+00, 1.24800E-01,
     *  1.62487E-02, 2.84237E-01,-1.68866E+00, 5.07723E-01, 5.14161E-01,
     * -4.71292E-01,-3.03487E+01, 4.17455E-01,-1.12591E+00, 0.00000E+00,
     * -3.03544E-01,-6.60313E-01,-1.48606E-01, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 1.00607E+01, 5.00000E+00/
C          UN2(5)
      DATA PBL1/
     *  0.00000E+00, 2.52207E+00, 3.84550E+00, 1.68023E+00, 7.93489E-01,
     *  3.93725E-02,-2.79707E+00,-4.76621E-01,-1.19972E-01, 3.20454E-01,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 4.17146E+00, 0.00000E+00,
     *  0.00000E+00,-5.30699E-01, 9.14373E-01, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00,-4.84434E-02, 1.85902E-01, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00/
      DATA PBL2/
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 5.00000E+00/
C          UN2(5)
      DATA PCL1/
     *  0.00000E+00, 1.55386E+01, 4.21418E+00,-9.70151E-01,-8.77326E-01,
     *  2.65813E-02, 1.40164E+00,-9.03874E-01, 3.17281E-03, 9.26891E+01,
     * -4.96004E+00, 0.00000E+00, 0.00000E+00,-4.17851E+00, 0.00000E+00,
     *  0.00000E+00,-1.14760E+01, 2.67744E+00, 0.00000E+00, 0.00000E+00,
     * -1.60056E+01, 0.00000E+00,-7.14647E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00,-2.89639E+00, 0.00000E+00, 0.00000E+00,-3.88601E+00,
     * -1.65784E+01, 8.44796E-01, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00,-3.75324E+00,-6.24047E+01, 0.00000E+00,
     *  0.00000E+00,-2.86808E-02,-1.95891E-01, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00,-3.10534E-01, 0.00000E+00,-3.37448E+00/
      DATA PCL2/
     *  1.63964E-02,-1.45191E+00, 1.85618E+00,-9.61979E-01, 3.93783E-01,
     *  4.21681E-01,-5.30254E-01,-2.96232E-01,-7.55211E-01, 0.00000E+00,
     * -1.85443E+00, 1.88047E+00,-1.07818E+00,-1.35373E+01,-3.05785E-01,
     *  7.82159E-01, 1.32586E+00, 2.34413E-01,-7.47152E-01, 9.92893E-01,
     * -2.80110E-02, 3.61747E-01,-4.16280E-01,-3.46427E-01,-5.76431E-01,
     * -2.13906E-01, 9.51184E-01, 3.69403E-01,-1.35563E+00, 6.59534E-02,
     *  1.39764E-01, 4.50687E-01,-1.22025E+00, 5.73280E-02, 7.49303E-01,
     * -8.37947E-01,-3.01332E+01, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     * -4.36697E-01,-7.76068E-01,-1.41680E-01, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 1.21958E+01, 5.00000E+00/
C          UN2(6)
      DATA PBM1/
     *  0.00000E+00, 3.13842E+00,-8.20417E-01, 3.72282E+00,-5.20477E-01,
     * -3.61867E-01,-2.92604E+00, 3.13013E-01,-1.38865E-01, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 1.30060E+01, 0.00000E+00,
     *  0.00000E+00, 1.67696E+00, 9.85990E-01, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00,-8.46922E-02, 5.59429E-03, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00/
      DATA PBM2/
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 5.00000E+00/
C          UN2(6)
      DATA PCM1/
     *  0.00000E+00, 1.78539E+01, 1.07314E+01,-1.13212E+00, 1.59867E-02,
     *  1.53736E-01, 2.25710E+00,-9.39080E-01,-9.72620E-02, 9.89789E+01,
     * -5.17469E+00, 0.00000E+00, 0.00000E+00,-2.98597E+00, 0.00000E+00,
     *  0.00000E+00,-2.04707E+01, 4.92899E+00, 0.00000E+00, 0.00000E+00,
     * -1.44316E+01, 0.00000E+00,-3.31557E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00,-6.22743E+00, 0.00000E+00, 0.00000E+00,-4.34344E+00,
     * -8.29640E+00,-3.03800E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 2.79387E+00,-5.23752E+01, 0.00000E+00,
     *  0.00000E+00,-2.59963E-02,-1.73426E-02, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00,-5.37220E+00, 0.00000E+00,-6.53478E-01/
      DATA PCM2/
     *  3.48181E-01,-1.88980E+00, 1.47787E+00,-7.92670E-01, 6.49224E-01,
     *  5.96079E-01,-1.04901E+00,-5.24003E-01,-6.77311E-01, 0.00000E+00,
     * -2.26873E+00, 2.80910E+00,-9.84994E-01,-6.79661E+00,-3.71975E-01,
     *  1.13310E+00, 1.57164E+00, 2.15176E-01,-5.58583E-01, 1.16045E+00,
     *  2.05395E-02, 2.27714E-01, 1.41203E-01,-3.92231E-01,-8.82859E-01,
     *  4.90400E-01, 1.14013E+00,-2.25250E-01,-1.64930E+00, 5.73434E-02,
     *  1.89857E-01, 4.31221E-01,-1.35345E+00,-2.94189E-01, 6.87530E-01,
     * -7.78284E-01,-2.88975E+01, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     * -3.98115E-01,-7.40699E-01,-8.28264E-02, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 2.02069E+00, 5.00000E+00/
C          UN2(7)
      DATA PBN1/
     *  0.00000E+00, 2.08818E+00,-1.96235E+00, 4.55317E+00,-1.76012E+00,
     * -4.75258E-01,-1.44220E+00,-3.28566E-01,-1.41177E-01, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 1.49146E+01, 0.00000E+00,
     *  0.00000E+00, 1.73222E+00, 9.91286E-01, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00,-1.35468E-01, 1.91833E-02, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00/
      DATA PBN2/
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 5.00000E+00/
C          UN2(7)
      DATA PCN1/
     *  0.00000E+00, 1.25645E+01, 2.43937E+01,-4.89691E-01,-5.46437E-01,
     *  1.22200E-01, 2.89309E+00,-2.85509E-01,-2.27122E-01, 9.54192E+01,
     * -4.07394E+00, 0.00000E+00, 0.00000E+00,-3.04354E+00, 0.00000E+00,
     *  0.00000E+00,-2.36547E+01, 1.04903E+01, 0.00000E+00, 0.00000E+00,
     * -8.32274E+00, 0.00000E+00,-3.34712E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00,-7.95953E+00, 0.00000E+00, 0.00000E+00,-5.83474E+00,
     * -1.48074E+00, 1.02268E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 6.19470E+00,-3.90767E+01, 0.00000E+00,
     *  0.00000E+00,-3.58136E-03, 1.22289E-03, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00,-8.49787E+00, 0.00000E+00,-3.97498E+00/
      DATA PCN2/
     *  3.79580E-01,-1.93595E+00, 2.89114E+00,-4.73457E-01, 7.67548E-01,
     *  5.66859E-01,-1.28683E+00,-8.37174E-01,-3.48022E-01, 0.00000E+00,
     * -2.62865E+00, 3.50575E+00,-7.93257E-01,-8.10692E-01,-4.99450E-01,
     *  1.56654E+00, 1.63039E+00, 7.58900E-02,-4.30952E-01, 1.23068E+00,
     *  1.06404E-01, 4.73870E-02, 5.50559E-01,-4.11375E-01,-9.94162E-01,
     *  1.35025E+00, 1.26053E+00,-7.34502E-01,-2.01952E+00, 2.05398E-01,
     * -4.77248E-02, 2.41549E-01,-9.32522E-01,-5.63663E-01, 5.34833E-01,
     * -5.77563E-01,-2.65033E+01, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     * -2.42317E-01,-7.33679E-01,-7.85537E-02, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 1.56842E-01, 5.00000E+00/
C          UN2(8)
      DATA PBO1/
     *  0.00000E+00, 7.00409E-01,-4.17017E-01, 3.24757E+00,-1.28352E+00,
     * -4.23875E-01, 1.64346E+00,-1.20855E+00,-7.65316E-01, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00,-3.39417E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 2.68534E+01, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00,-1.56444E-01,-4.60043E-02, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00/
      DATA PBO2/
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 5.00000E+00/
C          UN2(8)
      DATA PCO1/
     *  0.00000E+00, 7.30129E+00, 3.14811E+01,-7.06834E-02,-2.96193E-01,
     *  1.73817E-01, 1.62127E+00,-2.71556E-01,-2.05844E-01, 8.02088E+01,
     * -1.86956E-01, 0.00000E+00, 0.00000E+00,-9.43641E-01,-3.24716E+00,
     *  0.00000E+00,-2.32748E+01, 1.96724E+01, 0.00000E+00, 0.00000E+00,
     * -3.95949E+00, 0.00000E+00, 5.44787E-01, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00,-1.00161E+01, 0.00000E+00, 0.00000E+00,-4.57422E+00,
     *  4.31304E+00, 1.49868E+01, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 5.99489E+00,-2.82120E+01, 0.00000E+00,
     *  0.00000E+00, 4.03624E-02, 1.19463E-01, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00,-1.39050E+01, 0.00000E+00,-2.65634E+00/
      DATA PCO2/
     *  6.37036E-01,-1.77461E+00, 3.03103E+00,-1.49839E-01, 7.02027E-01,
     *  6.08841E-01,-9.27289E-01,-8.52362E-01, 5.61723E-01, 0.00000E+00,
     * -2.72061E+00, 3.66183E+00,-2.54943E-01, 2.94668E+00,-3.57898E-01,
     *  1.71858E+00, 1.58782E+00,-2.42995E-01,-3.57783E-01, 1.20157E+00,
     *  2.58895E-01,-1.05773E-01, 5.79397E-01,-3.30395E-01,-4.03569E-01,
     *  1.99175E+00, 1.21688E+00,-8.64350E-01,-1.95569E+00, 4.61136E-01,
     * -8.61382E-02, 3.38859E-01, 0.00000E+00,-5.78864E-01, 4.46659E-01,
     * -4.57428E-01,-1.99920E+01, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     * -1.19841E-01,-4.56968E-01, 2.00180E-02, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00,-1.07368E+00, 5.00000E+00/
C          UN2(9)
      DATA PBP1/
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 1.75863E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 3.18522E+01, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00/
      DATA PBP2/
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 5.00000E+00/
C          UN2(9)
      DATA PCP1/
     *  0.00000E+00, 4.61019E-02, 3.50615E+01, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 6.15349E+01,
     *  4.28634E+00, 0.00000E+00, 0.00000E+00, 6.03982E+00,-4.72305E+00,
     *  0.00000E+00,-1.43678E+01, 3.62580E+01, 0.00000E+00, 0.00000E+00,
     *  1.26574E+00, 0.00000E+00,-2.77285E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00,-1.14802E+01, 0.00000E+00, 0.00000E+00,-1.11940E+01,
     * -1.39535E+00, 2.63070E+01, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00,-1.53024E+00,-2.14609E+01, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00,-1.26956E+01, 0.00000E+00, 5.49926E+00/
      DATA PCP2/
     *  9.80142E-01,-1.19016E+00, 2.75110E+00, 4.23423E-01, 5.89893E-01,
     *  4.94288E-01,-5.25954E-01,-8.51760E-01, 1.62676E+00, 0.00000E+00,
     * -1.90027E+00, 3.19950E+00, 4.72739E-01, 7.04179E+00,-1.43685E-03,
     *  1.43219E+00, 1.32136E+00,-2.92744E-03,-3.43680E-01, 7.75735E-01,
     *  6.92202E-01,-1.45519E-01, 6.97813E-02,-3.11588E-01, 6.65750E-01,
     *  2.33809E+00, 1.06694E+00,-5.77590E-01,-1.33717E+00, 8.13367E-01,
     * -5.05737E-01, 5.99169E-01,-8.83386E-01,-4.38123E-01, 2.63649E-01,
     * -3.03448E-01,-1.28190E+01, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  1.45478E-02, 1.45491E-01, 2.40080E-01, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00,-3.86910E+00, 5.00000E+00/
C          UN2(10)
      DATA PBQ1/
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 1.10647E+01, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 3.13252E+01, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00/
      DATA PBQ2/
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 5.00000E+00/
C          UN2(10)
      DATA PCQ1/
     *  0.00000E+00,-3.03260E+00, 3.15488E+01, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 4.42798E+01,
     *  7.08849E+00, 0.00000E+00, 0.00000E+00, 1.64773E+01,-6.86505E+00,
     *  0.00000E+00,-6.27112E+00, 3.78373E+01, 0.00000E+00, 0.00000E+00,
     *  2.97763E+00, 0.00000E+00,-3.44134E-01, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00,-1.19424E+01, 0.00000E+00, 0.00000E+00,-1.64645E+01,
     * -2.27053E+00, 3.82330E+01, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 1.33140E-01,-2.08131E+01, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00,-7.04687E+00, 0.00000E+00, 6.52184E+00/
      DATA PCQ2/
     *  7.31799E-01,-2.75395E-01, 1.92467E+00, 8.71269E-01, 3.72836E-01,
     *  3.04967E-01, 7.72480E-02,-5.08596E-01, 1.99828E+00, 0.00000E+00,
     * -5.51169E-01, 2.12420E+00, 8.96069E-01, 1.12092E+01,-4.30438E-02,
     *  7.38391E-01, 6.12050E-01, 3.62981E-02,-1.02054E-01, 1.82404E-01,
     *  3.70643E-01,-1.68899E-01,-1.79628E-01,-1.21117E-01, 1.45823E+00,
     *  2.04352E+00, 7.83711E-01,-3.42979E-02,-2.31363E-01, 7.11253E-01,
     * -3.16353E-01, 6.21069E-01,-1.05676E+00,-4.03488E-01, 4.11595E-01,
     * -2.12535E-01,-6.51453E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  1.48238E-01, 6.38716E-01, 2.99311E-01, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00,-1.01846E+00, 5.00000E+00/
C          UN2(11)
      DATA PBR1/
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 2.21764E+01, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 6.77475E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00/
      DATA PBR2/
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 5.00000E+00/
C          UN2(11)
      DATA PCR1/
     *  0.00000E+00,-1.74115E+00, 2.66621E+01, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 3.13017E+01,
     *  6.86985E+00, 0.00000E+00, 0.00000E+00, 2.08835E+01,-7.86030E+00,
     *  0.00000E+00,-3.77141E+00, 3.87788E+01, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 1.31580E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00,-9.98927E+00, 0.00000E+00, 0.00000E+00,-1.71002E+01,
     * -9.88358E-01, 4.47756E+01, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 5.95029E-01,-2.11313E+01, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00,-3.84164E+00, 0.00000E+00, 0.00000E+00/
      DATA PCR2/
     *  3.07191E-01, 4.79094E-02, 6.72159E-01, 5.54185E-01, 1.82847E-01,
     * -1.23768E-02, 1.91637E-01,-2.89429E-02, 1.18297E+00, 0.00000E+00,
     *  2.37450E-01, 9.23551E-01, 6.05670E-01, 1.35990E+01,-1.64210E-01,
     *  5.38355E-03,-4.91246E-02,-1.06966E-01,-2.09635E-01,-3.23023E-02,
     * -3.41663E-02,-3.48871E-02,-2.62450E-01, 2.21492E-01, 1.43749E+00,
     *  1.08677E+00, 3.97778E-01, 3.61526E-01, 5.55950E-01, 3.53058E-01,
     * -5.93339E-02, 4.14203E-01,-6.05024E-01,-1.38714E-01, 2.78897E-01,
     * -8.92889E-02,-3.59033E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  9.90623E-02, 4.36170E-01, 7.95418E-02, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00,-1.11426E+00, 5.00000E+00/
C          UN2(12)
      DATA PBS1/
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 3.07320E+01, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 1.60738E+01, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00/
      DATA PBS2/
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 5.00000E+00/
C          UN2(12)
      DATA PCS1/
     *  0.00000E+00, 1.26217E+01, 2.30787E+01, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 2.00029E+01,
     * -2.88682E+00, 0.00000E+00, 0.00000E+00, 2.09439E+01,-4.56923E+00,
     *  0.00000E+00,-2.15929E+00, 3.87149E+01, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00,-7.98039E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00,-6.63423E+00, 0.00000E+00, 0.00000E+00,-5.84850E+00,
     *  3.72111E+00, 4.52300E+01, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 3.21872E-01, 0.00000E+00, 0.00000E+00/
      DATA PCS2/
     *  1.09405E-02,-4.35341E-02, 8.00586E-02, 1.48577E-01, 1.01602E-01,
     * -1.01104E-01,-1.98993E-02, 3.51174E-02, 2.41112E-01, 0.00000E+00,
     *  2.76479E-01, 1.97043E-01, 2.68708E-01, 1.39832E+01,-1.56638E-01,
     * -2.39101E-01,-1.50605E-01,-2.17139E-01,-2.59057E-01,-4.36362E-01,
     * -1.43496E-01, 7.51305E-02,-2.40850E-01, 1.34858E-01, 7.59193E-01,
     *  3.52708E-01, 1.29922E-01, 3.27957E-01, 5.35491E-01, 1.19120E-01,
     * -2.94029E-02, 1.76113E-01,-6.51597E-01, 3.61575E-02, 4.26836E-02,
     * -2.29297E-02,-4.27373E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     * -2.78548E-02, 5.77322E-02,-1.02411E-01, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 5.00000E+00/
C          UN2(13)
      DATA PBT1/
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 3.69447E+01, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 2.34073E+01, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00/
      DATA PBT2/
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 5.00000E+00/
C          UN2(13)
      DATA PCT1/
     *  0.00000E+00, 1.22096E+01, 1.92342E+01, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 8.13667E+00,
     * -6.19078E+00, 0.00000E+00, 0.00000E+00, 2.37009E+01, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00,-7.87365E+01, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00,-1.12371E+01, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00,-2.76047E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 1.85864E+01, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00/
      DATA PCT2/
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 5.00000E+00/
C          UN2(14)
      DATA PBU1/
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 1.01008E+01, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 2.21469E+01, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00/
      DATA PBU2/
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 5.00000E+00/
C          UN2(14)
      DATA PCU1/
     *  0.00000E+00,-1.40697E+00, 6.88709E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 3.67624E+02, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 1.58312E+01, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00,-2.46486E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00,-1.90327E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 1.13248E+01, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00/
      DATA PCU2/
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00,
     *  0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 5.00000E+00/
      END
