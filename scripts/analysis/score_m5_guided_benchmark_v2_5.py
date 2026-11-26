"""Score v2.5 with the v2.4 independent optical and structural criteria."""
import score_m5_guided_benchmark_v2_4 as base

ID = 'm5_guided_benchmark_v2_5'
SPEC = base.ROOT/f'config/task/{ID}.json'
registration = base.bind(base.registration, ID=ID, __file__=__file__)
predict = base.bind(base.predict, SPEC=SPEC, registration=registration)
score = base.bind(base.score, SPEC=SPEC, registration=registration)
main = base.bind(base.main, __doc__=__doc__, SPEC=SPEC, registration=registration,
                 predict=predict, score=score)

if __name__=='__main__':
    main()
